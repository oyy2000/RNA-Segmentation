#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../..')

import argparse
import os
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Ensure you have the 'fm' module. If not, clone the RNA-FM repository:
# git clone https://github.com/chenyifanbio/RNA-FM.git
import fm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Segmentation Model with optional CRF and Pseudo Labeling.')
parser.add_argument('--use_crf', action='store_true', help='Use CRF in the model.')
parser.add_argument('--pseudo_labeling', action='store_true', help='Perform Pseudo Labeling.')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load RNA-FM backbone and alphabet
backbone, alphabet = fm.pretrained.rna_fm_t12()
backbone.to(device)
backbone.eval()
print("Created RNA-FM backbone successfully")

class Human_5Prime_UTR_VarLength(object):
    def __init__(self, root):
        self.root = root
        self.src_csv_path = os.path.join(self.root, "data", "GSM4084997_varying_length_25to100_with_cams.csv")
        self.seqs, self.scaled_rls = self.__dataset_info(self.src_csv_path)

    def __getitem__(self, index):
        seq_str = self.seqs[index].replace("T", "U")
        label = self.scaled_rls[index]
        return seq_str, label

    def __len__(self):
        return len(self.seqs)

    def __dataset_info(self, src_csv_path):
        # Load and process the data
        src_df = pd.read_csv(src_csv_path)
        src_df.loc[:, "ori_index"] = src_df.index
        random_df = src_df[src_df['set'] == 'random']
        random_df = random_df[random_df['total_reads'] >= 10]
        random_df.reset_index(drop=True, inplace=True)

        human_df = src_df[src_df['set'] == 'human']
        human_df = human_df[human_df['total_reads'] >= 10]
        human_df.reset_index(drop=True, inplace=True)

        combined_df = pd.concat([random_df, human_df], ignore_index=True)

        label_col = 'rl'
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(combined_df[label_col].values.reshape(-1, 1))
        combined_df['scaled_rl'] = self.scaler.transform(combined_df[label_col].values.reshape(-1, 1))

        seqs = combined_df['utr'].values
        scaled_rls = combined_df['scaled_rl'].values 

        print(f"Number of samples in dataset: {len(seqs)}")
        return seqs, scaled_rls

def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):              
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str) + int(alphabet.prepend_bos)] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens

def collate_fn(batch):
    seq_strs, labels = zip(*batch)
    tokens = generate_token_batch(alphabet, seq_strs)
    labels = torch.Tensor(labels)
    lengths = [len(seq_str) + int(alphabet.prepend_bos) + int(alphabet.append_eos) for seq_str in seq_strs]
    return seq_strs, tokens, labels, lengths

# Initialize Dataset and DataLoader
root_path = "./"
dataset = Human_5Prime_UTR_VarLength(root=root_path)

batch_size = 64
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, collate_fn=collate_fn, drop_last=False
)

scaler = dataset.scaler

# Define Models
if args.use_crf:
    # Install torchcrf if not already installed
    # !pip install pytorch-crf
    from torchcrf import CRF

    class SegmentationModelWithCRF(nn.Module):
        def __init__(self, vocab_size, embed_dim=64):
            super(SegmentationModelWithCRF, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.encoder = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
            self.hidden2tag = nn.Linear(256, 2)  # For binary classification
            self.crf = CRF(num_tags=2, batch_first=True)

        def forward(self, x, lengths):
            x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x_packed, _ = self.encoder(x_packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
            emissions = self.hidden2tag(x)  # [batch_size, seq_len, num_tags]
            return emissions

        def loss(self, emissions, tags, mask):
            return -self.crf(emissions, tags, mask=mask)

        def decode(self, emissions, mask):
            return self.crf.decode(emissions, mask=mask)

    model = SegmentationModelWithCRF(vocab_size=len(alphabet.all_toks)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    class SequenceRegressionModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=64):
            super(SequenceRegressionModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.encoder = nn.LSTM(embed_dim, hidden_size=128, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(256, 1)
            
        def forward(self, x, lengths):
            x = self.embedding(x)
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.encoder(x_packed)
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            out = self.fc(h_n)
            return out.squeeze(1)

    model = SequenceRegressionModel(vocab_size=len(alphabet.all_toks)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (seq_strs, sequences, labels, lengths) in enumerate(data_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = torch.tensor(lengths, dtype=torch.long).to(device)
        optimizer.zero_grad()

        if args.use_crf:
            emissions = model(sequences, lengths)
            # Generate some random tags for example purposes
            tags = torch.randint(0, 2, emissions.shape[:2], dtype=torch.long).to(device)
            mask = sequences != alphabet.padding_idx
            loss = model.loss(emissions, tags, mask)
        else:
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Pseudo Labeling
if args.pseudo_labeling:
    print("Starting Pseudo Labeling...")
    # Assuming we have an unlabeled dataset called 'unlabeled_dataset'
    # For demonstration, we'll reuse the same dataset
    unlabeled_dataset = dataset
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False
    )

    model.eval()
    new_pseudo_labels = []
    with torch.no_grad():
        for seq_strs, sequences, _, lengths in unlabeled_loader:
            sequences = sequences.to(device)
            lengths = torch.tensor(lengths, dtype=torch.long).to(device)
            if args.use_crf:
                emissions = model(sequences, lengths)
                mask = sequences != alphabet.padding_idx
                predicted_labels = model.decode(emissions, mask)
                # Flatten and collect labels
                new_pseudo_labels.extend(predicted_labels)
            else:
                outputs = model(sequences, lengths)
                predicted_labels = outputs.cpu().numpy()
                new_pseudo_labels.extend(predicted_labels)

    # Update Dataset with New Pseudo Labels
    # In practice, you should filter or select pseudo labels based on confidence
    # Here, we'll just replace the labels with pseudo labels
    if args.use_crf:
        # For CRF, the labels are sequences
        # You need to process them accordingly
        pass  # Implement processing if needed
    else:
        # For regression, update the labels
        dataset.scaled_rls = np.array(new_pseudo_labels)

    # Retrain the Model with New Pseudo Labels
    print("Retraining with Pseudo Labels...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (seq_strs, sequences, labels, lengths) in enumerate(data_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = torch.tensor(lengths, dtype=torch.long).to(device)
            optimizer.zero_grad()

            if args.use_crf:
                emissions = model(sequences, lengths)
                tags = torch.randint(0, 2, emissions.shape[:2], dtype=torch.long).to(device)
                mask = sequences != alphabet.padding_idx
                loss = model.loss(emissions, tags, mask)
            else:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"[Pseudo Labeling] Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")