import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
import numpy as np
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import random


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Parsing the input file
def parse_file(file_path, window_size=15):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    targets = []
    sequences = []

    for i in range(0, len(lines), 3):
        sequence = lines[i+1].strip()  # Sequence
        target = lines[i+2].strip()  # Target

        sequences.append(sequence)
        targets.append([int(x) for x in target])

    return sequences, targets


def find_metrics(y_predict, y_proba, y_test):

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:])
    auPR = average_precision_score(y_test, y_proba[:])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


# Define the RNN Model
class ResidueCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidueCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = output_size

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.fc = nn.Linear(32*1280, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_size)
        
    def forward(self, x):
        # Get the current batch size
        current_batch_size = x.size(0)

        # Pad the input if necessary
        if current_batch_size < self.batch_size:
            padding = torch.zeros(self.batch_size - current_batch_size, *x.shape[1:], device=x.device)
            x = torch.cat([x, padding], dim=0)

        x = x.T
        # Add a channel dimension
        x = x.unsqueeze(0)

        out = self.conv1d(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)

        out = self.fc2(out)

        # out = torch.sigmoid(out)
        out = out.view(-1)

        if current_batch_size < self.batch_size:
            out = out[:current_batch_size]
        return out


# Hyperparameters
learning_rate = 0.0001
epochs = 20
hidden_size = 128  # Size of the LSTM hidden layer
num_layers = 2  # Number of LSTM layers

# Load data
file_path = 'TS88.txt'  # Replace with your file path
sequences, targets = parse_file(file_path)

all_targets = []
for i in targets:
    for j in i:
        all_targets.append(j)

all_targets = np.array(all_targets)
print(Counter(all_targets))

X_test = pd.read_csv('TS88_embeddings.csv', header=None).values
X_test = X_test[:, 1:]

# Model, loss, and optimizer
input_size = 1024 
output_size = 1280
model = ResidueCNN(input_size=input_size, output_size=output_size)

# load the model
model.load_state_dict(torch.load('ResidueCNN.pth'))
model.eval()

prev = 0
global_y_proba = []
global_y_predict = []
global_y_test = []
for i in range(len(sequences)):
    # Get the sequence
    x = torch.tensor(X_test[prev:prev+len(sequences[i])], dtype=torch.float32)

    # Get the target
    y = torch.tensor(targets[i], dtype=torch.float32)

    prev += len(sequences[i])

    # Forward pass
    y_pred = model(x)
    y_proba = torch.sigmoid(y_pred)
    y_proba = y_proba.detach().numpy()

    y_predict = (y_proba > 0.55).astype(int)

    global_y_proba.extend(y_proba)
    global_y_predict.extend(y_predict)
    global_y_test.extend(y)


global_y_proba = np.array(global_y_proba)
global_y_predict = np.array(global_y_predict)
global_y_test = np.array(global_y_test)

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(global_y_predict, global_y_proba, global_y_test)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Balanced Accuracy:', bal_acc)
print('Accuracy:', acc)
print('Precision:', prec)
print('F1 Score:', f1_score_1)
print('MCC:', mcc)
print('AUC:', auc)
print('auPR:', auPR)


