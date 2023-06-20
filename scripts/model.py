import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from losses import ConfidenceLoss


class ScorePredictor(nn.Module):
    def __init__(self, feature_dim):
        super(ScorePredictor, self).__init__()
        self.lin1 = nn.Linear(feature_dim, feature_dim * 4)
        self.act1 = nn.Sigmoid()
        self.lin2 = nn.Linear(feature_dim * 4, 1)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        return x


class ScorePredictorResidual(nn.Module):
    def __init__(self, feature_dim):
        super(ScorePredictor, self).__init__()
        self.lin1 = nn.Linear(feature_dim, feature_dim * 2)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(feature_dim * 2, feature_dim)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(feature_dim, 1)
        self.act3 = nn.Sigmoid()
    
    def forward(self, x):
        inp = x
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x)) + inp
        x = self.act3(self.lin3(x))
        return x


def train_score_predictor(
    train_features, 
    softmax_output, 
    target_distribution, 
    num_epochs, 
    batch_size, 
    ):
    feature_dim = train_features.shape[1]
    N_train = train_features.shape[0]

    model = ScorePredictor(feature_dim=feature_dim)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = ConfidenceLoss(scaling_factor=0.01) 


    hist = []
    device = "cpu"
    num_batches = N_train // batch_size

    model = model.to(device)

    model.train()
    for t in range(num_epochs):
        loss_epoch = 0
        for i in range(0, N_train, batch_size):
            optimizer.zero_grad()

            p_batch = torch.tensor(softmax_output[i:i+batch_size]).to(device)
            y_batch = torch.tensor(target_distribution[i:i+batch_size]).to(device)
            features_batch = torch.tensor(train_features[i:i+batch_size], dtype=torch.float32).to(device)
            c_batch = model(features_batch)

            loss = loss_fn(p_batch, y_batch, c_batch)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        print("epoch: ", t, " average_loss: ", loss_epoch / num_batches)
    return model


def test_score_predictor(
    model, 
    features, 
    true_labels, 
    predicted_labels, 
    batch_size, 
    plot_rejection_curve
):

    N = features.shape[0]
    step = batch_size
    device = "cpu"
    # getting scores
    model.eval()
    scores = np.zeros(N)

    with torch.no_grad():
        for i in range(step, N, step):
            features_batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
            c_batch = model(features_batch)
            scores[i:i+batch_size] = c_batch.flatten().detach().cpu()

    # get accuracy rejection curve for evaluating scores
    r_rate = [0]
    true_labels = true_labels
    r_accuracy = [accuracy_score(predicted_labels, true_labels)]
    max_accuracy = [accuracy_score(predicted_labels, true_labels)]
    step = batch_size
    idx = np.argsort(scores)[::-1]

    for i in range (step, N, step):
        idx = idx[:(N - i)]
        r_rate.append(i / N)
        r_accuracy.append(accuracy_score(predicted_labels[idx], true_labels[idx]))

    if plot_rejection_curve:    
        plt.plot(r_rate, r_accuracy, label="topological estimator")
    
        plt.xlabel("rejection rate")
        plt.ylabel("accuracy")
        plt.title("UE via topological featues")
        plt.legend()
        plt.show()

    # calculate area under rejection curve
    s0 = r_accuracy[0]
    s = np.trapz(r_accuracy, dx=step / N)
    return s - s0, scores
