import pandas as pd
import numpy as np
import torch
import shap
from sklearn.metrics import accuracy_score


def get_ideal_rejection(accuracy0, N_samples, step):
    total_accuracy = [accuracy0]
    for i in range(step, N_samples, step):
        N_new = N_samples - i
        accuracy_new = accuracy0 * N_samples / N_new
        total_accuracy.append(min(accuracy_new, 1.0))
    return total_accuracy


def get_accuracy_rejection(true_labels, predicted_labels, probs, step):
    r_rate = [0]
    N = len(predicted_labels)
    idx = np.argsort(probs)[::-1]
    val_idx = idx
    total_accuracy = [accuracy_score(predicted_labels, true_labels)]
    for i in range (step, N, step):
        idx = idx[:(N - i)]
        r_rate.append(i / N)
        total_accuracy.append(accuracy_score(predicted_labels[idx], true_labels[idx]))
    return r_rate, total_accuracy


def get_target_distribution(train_labels):
    N_train = len(train_labels)
    target_distribution = np.zeros((N_train, 2))
    for i in range(N_train):
        target_distribution[i, train_labels[i]] = 1.0
    return target_distribution


def get_shap_values(model, X_train, X_test):
    device = "cpu"
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    return shap_values
