#!/usr/bin/env python
# coding: utf-8

# Import necessary modules
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../..')

import os
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import fm
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(2021)

# ## 1. Load Model

# ### (1) Define Human5PrimeUTRPredictor

class Human5PrimeUTRPredictor(torch.nn.Module):
    def __init__(self, alphabet=None, task="rgs", arch="cnn", input_types=["seq", "emb-rnafm"]):
        super().__init__()
        self.alphabet = alphabet
        self.task = task
        self.arch = arch
        self.input_types = input_types
        self.padding_mode = "right"
        self.token_len = 100
        self.out_plane = 1
        self.in_channels = 0
        if "seq" in self.input_types:
            self.in_channels += 4

        if "emb-rnafm" in self.input_types:
            self.reductio_module = nn.Linear(640, 32)
            self.in_channels += 32  

        if self.arch == "cnn" and self.in_channels != 0:
            self.predictor = CNNModel(in_planes=self.in_channels, out_planes=1)
        else:
            raise Exception("Wrong Arch Type")

    def forward(self, tokens, inputs):
        ensemble_inputs = []
        if "seq" in self.input_types:
            # Padding one-hot embedding            
            nest_tokens = (tokens[:, 1:-1] - 4)   # Convert tokens for RNA-FM to nest version
            nest_tokens = torch.nn.functional.pad(nest_tokens, (0, self.token_len - nest_tokens.shape[1]), value=-2)
            token_padding_mask = nest_tokens.ge(0).long()
            one_hot_tokens = torch.nn.functional.one_hot((nest_tokens * token_padding_mask), num_classes=4)
            one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)            
            one_hot_tokens = one_hot_tokens.permute(0, 2, 1)  # B, 4, L
            ensemble_inputs.append(one_hot_tokens)

        if "emb-rnafm" in self.input_types:
            embeddings = inputs["emb-rnafm"]
            # Padding RNA-FM embedding
            embeddings, padding_masks = self.remove_pend_tokens_1d(tokens, embeddings)
            batch_size, seqlen, hiddendim = embeddings.size()
            embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, self.token_len - embeddings.shape[1]))            
            embeddings = self.reductio_module(embeddings)
            embeddings = embeddings.permute(0, 2, 1)
            ensemble_inputs.append(embeddings)        

        ensemble_inputs = torch.cat(ensemble_inputs, dim=1)        
        output, feature_maps = self.predictor(ensemble_inputs)
        output = output.squeeze(-1)
        return output, feature_maps

    def remove_pend_tokens_1d(self, tokens, seqs):
        padding_masks = tokens.ne(self.alphabet.padding_idx)

        # Remove eos token
        if self.alphabet.append_eos:
            eos_masks = tokens.ne(self.alphabet.eos_idx)
            eos_pad_masks = (eos_masks & padding_masks).to(seqs)
            seqs = seqs * eos_pad_masks.unsqueeze(-1)
            seqs = seqs[:, :-1, :]
            padding_masks = padding_masks[:, :-1]

        # Remove bos token
        if self.alphabet.prepend_bos:
            seqs = seqs[:, 1:, :]
            padding_masks = padding_masks[:, 1:]

        if not padding_masks.any():
            padding_masks = None

        return seqs, padding_masks

class CNNModel(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CNNModel, self).__init__()
        main_planes = 64
        dropout = 0.2

        self.conv1 = nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(main_planes, main_planes, stride=2, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.resblock2 = ResBlock(main_planes, main_planes, stride=1, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.resblock3 = ResBlock(main_planes, main_planes, stride=2, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.resblock4 = ResBlock(main_planes, main_planes, stride=1, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.resblock5 = ResBlock(main_planes, main_planes, stride=2, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.resblock6 = ResBlock(main_planes, main_planes, stride=1, conv_layer=nn.Conv1d,
                                  norm_layer=nn.BatchNorm1d)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(main_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        feature_maps = x  # Save feature maps before pooling
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.fc(x)
        return out, feature_maps

class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        conv_layer=nn.Conv1d,
        norm_layer=nn.BatchNorm1d,
    ):
        super(ResBlock, self).__init__()        
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)       
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, padding=1, bias=False)

        if stride > 1 or out_planes != in_planes: 
            self.downsample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None
            
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

# ### (2) Create RNA-FM Backbone

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone, alphabet = fm.pretrained.rna_fm_t12()
backbone.to(device)
backbone.eval()
print("Created RNA-FM backbone successfully")

# ### (3) Create UTR Function Downstream Predictor

task = "rgs"
arch = "cnn"
input_items = ["seq", "emb-rnafm"]
model_name = arch.upper() + "_" + "_".join(input_items) 
utr_func_predictor = Human5PrimeUTRPredictor(
    alphabet, task=task, arch=arch, input_types=input_items    
)
utr_func_predictor.to(device)

# Load the trained model
model_path = "result/{}_best_utr_predictor_epoch_14.pth".format(model_name)
utr_func_predictor.load_state_dict(torch.load(model_path, map_location=device))
utr_func_predictor.eval()
print("Loaded trained UTR function predictor successfully")

# ## 2. Load Data

# Read the original CSV file
root_path = "./"
src_csv_path = os.path.join(root_path, "data", "GSM4084997_varying_length_25to100.csv")
src_df = pd.read_csv(src_csv_path)

# Filter sequences with 'total_reads' >= 10
src_df = src_df[src_df['total_reads'] >= 10]
src_df.reset_index(drop=True, inplace=True)

# Replace 'T' with 'U' in sequences
src_df['utr'] = src_df['utr'].str.replace('T', 'U')

# Get sequences and labels
seqs = src_df['utr'].values
labels = src_df['rl'].values

# Define the Dataset class
class UTRDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, alphabet):
        self.seqs = seqs
        self.labels = labels
        self.alphabet = alphabet
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq_str = self.seqs[idx]
        label = self.labels[idx]
        return seq_str, label

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
    return seq_strs, tokens, labels

# Create the dataset and DataLoader
dataset = UTRDataset(seqs, labels, alphabet)
batch_size = 64
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False,
    num_workers=0, collate_fn=collate_fn, drop_last=False
)

# ## 3. Generate CAM Scores and Pseudo Labels

def compute_cams(feature_maps, model):
    # feature_maps: Tensor [batch_size, num_channels, sequence_length]
    fc_weights = model.predictor.fc.weight  # Shape: [1, num_channels]
    fc_weights = fc_weights.unsqueeze(-1)  # Shape: [1, num_channels, 1]
    cams = torch.mul(feature_maps, fc_weights).sum(dim=1)  # Shape: [batch_size, sequence_length]
    return cams

def upsample_cam(cam, seq_length):
    cam_length = len(cam)
    cam_positions = np.arange(cam_length)
    seq_positions = np.linspace(0, cam_length - 1, num=seq_length)
    cam_upsampled = np.interp(seq_positions, cam_positions, cam)
    return cam_upsampled

all_cams = []
all_pseudo_labels = []

utr_func_predictor.eval()
backbone.eval()

for seq_strs, tokens, labels in tqdm(data_loader):
    tokens = tokens.to(device)
    with torch.no_grad():
        inputs = {}
        if "emb-rnafm" in input_items:
            backbone_results = backbone(tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
            inputs["emb-rnafm"] = backbone_results["representations"][12]
        outputs, feature_maps = utr_func_predictor(tokens, inputs)
        cams = compute_cams(feature_maps, utr_func_predictor)
        cams_np = cams.cpu().numpy()
        # Process each sample in the batch
        for i in range(len(seq_strs)):
            seq_len = len(seq_strs[i])
            cam = cams_np[i][:feature_maps.size(-1)]
            cam = cam.squeeze()
            cam_upsampled = upsample_cam(cam, seq_len)
            # Normalize CAM
            if cam_upsampled.max() != cam_upsampled.min():
                cam_norm = (cam_upsampled - cam_upsampled.min()) / (cam_upsampled.max() - cam_upsampled.min())
            else:
                cam_norm = cam_upsampled
            # Generate pseudo label
            threshold = 0.5
            important_positions = (cam_norm >= threshold).astype(int)
            # Convert cam_norm and important_positions to strings
            cam_str = ';'.join(map(str, cam_norm))
            pseudo_label_str = ';'.join(map(str, important_positions))
            all_cams.append(cam_str)
            all_pseudo_labels.append(pseudo_label_str)

# Add CAMs and pseudo labels to the dataframe
src_df['cam'] = all_cams
src_df['pseudo_label'] = all_pseudo_labels

# Write the dataframe to a CSV file with selected columns
output_csv_path = 'GSM4084997_varying_length_25to100_with_cams.csv'
src_df[['utr', 'set', 'total_reads','rl', 'cam', 'pseudo_label']].to_csv(output_csv_path, index=False)

print(f"CAM scores and pseudo labels saved to {output_csv_path}")
