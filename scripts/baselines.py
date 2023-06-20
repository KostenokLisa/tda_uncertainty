import pandas as pd
import numpy as np
import torch.nn as nn
import tqdm

from tqdm import tqdm
from torch.optim import Adam
from torch.nn.functional import dropout


"""Implemented methods reuse functions from the code provided by authors of the paper
"Uncertainty Estimation of Transformer Predictions for Misclassification Detection"
https://github.com/AIRI-Institute/uncertainty_transformers
"""

class UEconfig:
    def __init__(self, ue_type, use_cache):
        self.ue_type = ue_type
        self.use_cache = use_cache
        self.dropout_type = 'MC'
        self.inference_prob = 0.1
        self.committee_size = 10
        self.dropout_subs = 'all'
        self.eval_passes = False
        self.calibrate = False
        self.use_selective = False


def create_ue_estimator(
    model,
    ue_args,
    eval_metric,
    calibration_dataset,
    train_dataset,
    cache_dir,
    config=None,
):
    if ue_args.ue_type == "mc" or ue_args.ue_type == "mc-dc":
        return UeEstimatorMc(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )
        raise ValueError()


def estimate(
    config,
    classifier,
    eval_metric,
    calibration_dataset,
    train_dataset,
    eval_dataset,
    eval_results,
    work_dir
):
    """Function for uncertainty estimation"""
    true_labels = eval_results["true_labels"]
    # create estimator
    ue_estimator = create_ue_estimator(
        classifier,
        config.ue,
        eval_metric,
        calibration_dataset=calibration_dataset,
        train_dataset=train_dataset,
        cache_dir=config.cache_dir,
        config=config,
    )
    # calc UE
    ue_results = ue_estimator(eval_dataset, true_labels)
    # save results
    eval_results.update(ue_results)
    print(ue_results)

"""Monte-Carlo Dropout utilities"""

class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return dropout(
            x, self.p, training=self.training or self.activate
        )
    

def convert_dropouts(model, ue_args):
    """This function replace all model dropouts with custom dropout layer."""
    dropout_ctor = lambda p, activate: DropoutMC(
        p=ue_args.inference_prob, activate=False
    )
    convert_to_mc_dropout(model, {"Dropout": dropout_ctor, "StableDropout": dropout_ctor})


def convert_to_mc_dropout(
    model, substitution_dict
):
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        proba_field_name = "drop_prob" if layer_name == "StableDropout" else proba_field_name #DeBERTA case
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)


def activate_mc_dropout(
    model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(
                model=layer, activate=activate, random=random, verbose=verbose
            )


def softmax(logits):
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


class UeEstimatorMc:
    """Monte-Carlo Dropout base method"""
    def __init__(self, ue_args, eval_metric, calibration_dataset, train_dataset):
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.train_dataset = train_dataset

    def __call__(self, model, eval_dataset, eval_mask, true_labels=None):
        ue_args = self.ue_args
        eval_metric = self.eval_metric


        convert_dropouts(model, ue_args)
        activate_mc_dropout(model, activate=True, random=ue_args.inference_prob)

        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        
        batch_size = 16
        N = eval_dataset.shape(0)
        predicted_labels = np.zeros(N)
        probs = np.zeros(N)
        for j in tqdm(np.arange(ue_args.committee_size)):
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    x_batch = torch.tensor(eval_dataset[i:i+batch_size]).to(device)
                    mask_batch = torch.tensor(eval_mask[i:i+batch_size]).to(device)

                    logits_batch = model(x_batch, attention_mask=mask_batch).logits
                    probs_batch = softmax(logits_batch)
                    pred_probs_batch, pred_labels_batch = torch.max(probs_batch, dim=1)
                    pred_probs_batch = pred_probs_batch.detach().cpu().numpy()
                    pred_labels_batch = pred_labels_batch.detach().cpu().numpy()
                    probs[i:i+batch_size] = pred_probs_batch
                    predicted_labels[i:i+batch_size] = pred_labels_batch

            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(predicted_labels.tolist())

        activate_mc_dropout(model, activate=False)

        return eval_results
    

def calc_score(probs, mode):
    if mode == "SMP":
        score = np.mean(probs, axis=1)
    elif mode == "PV":
        score = 1.0 - np.var(probs, axis=1)
    elif mode == "entropy":
        score = -np.sum(probs * np.log(probs), axis=1)
    return score

"""Malahanobis distance utilities"""

class BERTClassificationHeadIdentityPooler(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, model):
        super().__init__()
        self.pooler = model.bert.pooler

    def forward(self, features):
        x = features[:, 0, :]
        x = self.pooler(features)
        print(x.shape)
        return x


class BERTClf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return features
    

def compute_centroids(train_features, train_labels):
    centroids = []
    for label in np.sort(np.unique(train_labels)):
        print(train_features.shape)
        print(train_labels.shape)
        centroids.append(train_features[train_labels == label].mean(axis=0))
    return np.asarray(centroids)


def compute_covariance(centroids, train_features, train_labels):
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    for c, mu_c in tqdm(enumerate(centroids)):
        for x in train_features[train_labels == c]:
            d = (x - mu_c)[:, None]
            cov += d @ d.T
    return cov / train_features.shape[0]


def mahalanobis_distance(train_features, train_labels, eval_features):
    centroids = compute_centroids(train_features, train_labels)
    sigma = compute_covariance(centroids, train_features, train_labels)
    diff = eval_features[:, None, :] - centroids[None, :, :]
    try:
        sigma_inv = np.linalg.inv(sigma)
    except:
        sigma_inv = np.linalg.pinv(sigma)
    dists = np.matmul(np.matmul(diff, sigma_inv), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])
    return np.min(dists, axis=1)


class TextClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, data, mask, apply_softmax=False, return_preds=False):
        batch_size = 16
        N = data.shape[0]
        emb_dim = 768
        predicted_labels = np.zeros(N)
        probs = np.zeros((N, emb_dim))
        with torch.no_grad():
            for i in range(0, N, batch_size):
                x_batch = torch.tensor(data[i:i+batch_size]).to(device)
                mask_batch = torch.tensor(mask[i:i+batch_size]).to(device)

                logits_batch = model(x_batch, attention_mask=mask_batch).logits
                if apply_softmax:
                    probs_batch = softmax(logits_batch)
                else:
                    probs_batch = logits_batch
                if return_preds:
                    pred_probs_batch, pred_labels_batch = torch.max(probs_batch, dim=1)
                    pred_labels_batch = pred_labels_batch.detach().cpu().numpy()
                    predicted_labels[i:i+batch_size] = pred_labels_batch

                probs_batch = probs_batch.detach().cpu().numpy()
                probs[i:i+batch_size] = probs_batch

        if return_preds:
            return (probs, predicted_labels)
        return probs


class UeEstimatorMahalanobis:
    def __init__(self, ue_args, config, train_dataset):
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, model, val_data, val_mask, val_labels, train_data, train_mask, train_labels, true_labels=None):
        # change head
        self.cls = TextClassifier(model)
        self.cls.model.bert.pooler = BERTClassificationHeadIdentityPooler(model)
        self.cls.model.classifier = BERTClf()

        eval_features = self.cls.predict(
            val_data, val_mask, apply_softmax=False, return_preds=False
        )

        train_features = self.cls.predict(
            train_data, train_mask, apply_softmax=False, return_preds=False
        )

        eval_results = {}
        eval_results["eval_labels"] = true_labels
        eval_results["mahalanobis_distance"] = mahalanobis_distance(
            train_features, train_labels, eval_features
        ).tolist()

        return eval_results
