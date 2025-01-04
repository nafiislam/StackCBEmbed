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
import torch.nn.functional as F


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
    auc = roc_auc_score(y_test, y_proba[:, :])
    auPR = average_precision_score(y_test, y_proba[:, :])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


# Define the RNN Model
class ResidueTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads, max_seq_len):
        super(ResidueTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.batch_size = 1280

        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        current_batch_size = x.size(0)

        # Pad the input if necessary
        if current_batch_size < self.batch_size:
            padding = torch.zeros(self.batch_size - current_batch_size, *x.shape[1:], device=x.device)
            x = torch.cat([x, padding], dim=0)

        x = x.unsqueeze(0)  # Add a batch dimension
        # Input embedding
        x = self.input_embedding(x)

        x = x + self.positional_encoding

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Take the output from the last time step
        out = x[:, -1, :]  # Shape: [batch_size, hidden_size]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        # Squeeze to remove extra dimensions
        out = out.view(-1)

        if current_batch_size < self.batch_size:
            out = out[:current_batch_size]
        return out


# Hyperparameters
learning_rate = 0.0001
epochs = 40
hidden_size = 512  # Size of the LSTM hidden layer
num_layers = 2  # Number of LSTM layers

# Load data
file_path = 'Benchmark.txt'  # Replace with your file path
sequences, targets = parse_file(file_path)

all_targets = []
for i in targets:
    for j in i:
        all_targets.append(j)

all_targets = np.array(all_targets)
print(Counter(all_targets))

X_train = pd.read_csv('benchmark_embeddings.csv', header=None).values
X_train = X_train[:, 1:]

# Model, loss, and optimizer
input_size = 1024 
output_size = 1280
model = ResidueTransformer(input_size, hidden_size, num_layers=4, output_size=output_size, num_heads=8, max_seq_len=1280)

# Calculate class frequencies
num_positive = (all_targets == 1).sum()
num_negative = (all_targets == 0).sum()
pos_weight = torch.tensor([1000*num_negative / num_positive], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train by sequence by sequence
for ep in range(epochs):
    prev = 0
    train_loss = 0
    for i in range(len(sequences)):
        # Get the sequence
        x = torch.tensor(X_train[prev:prev+len(sequences[i])], dtype=torch.float32)

        # Get the target
        y = torch.tensor(targets[i], dtype=torch.float32)

        prev += len(sequences[i])

        # Forward pass
        y_pred = model(x)

        # Calculate loss
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # print(f'Epoch [{ep+1}/{epochs}], Sequence [{i+1}/{len(sequences)}], Loss: {loss.item()}')
    
    print(f'Epoch [{ep+1}/{epochs}], Loss: {train_loss/len(sequences)}')

# save the model
torch.save(model.state_dict(), 'ResidueTransformer.pth')