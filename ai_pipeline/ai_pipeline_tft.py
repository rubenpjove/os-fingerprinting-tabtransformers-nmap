import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import absl.logging
import warnings
import logging

import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer
from torchmetrics import AUROC
from tqdm import tqdm
import torch.optim as optim

###############################

logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)

seed = 2024
np.random.seed(seed)

df = pd.read_csv("./dataset_no_encoded_4397.csv")

df.pop('Class.vendor_0')
df.pop('Class.OSgen_0')
df.pop('Class.device_0')
df.reset_index(drop=True, inplace=True)

OutVar = list(df.columns)[0]

df.drop_duplicates(keep=False, inplace=True)

# df.replace(['BSD', 'iOS', 'macOS', 'Solaris', 'Android'], 'Other', inplace=True)
df = df[~df.isin(['BSD', 'iOS', 'macOS', 'Solaris', 'Android']).any(axis=1)]
df.reset_index(drop=True, inplace=True)

###############################

LABEL = OutVar

NUMERIC_FEATURES = df.select_dtypes(include=['int64']).columns.tolist()
CATEGORICAL_FEATURES = df.select_dtypes(include=['object']).columns.tolist()
CATEGORICAL_FEATURES.remove(LABEL)

FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

train_data, test_data = train_test_split(df, stratify=df[LABEL], test_size=0.20, random_state=seed)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
numeric_pipe = Pipeline([
    ('impute', imputer),
    ('scale', scaler),
])
numeric_pipe.fit(train_data[NUMERIC_FEATURES])
train_data[NUMERIC_FEATURES] = numeric_pipe.transform(train_data[NUMERIC_FEATURES])
test_data[NUMERIC_FEATURES] = numeric_pipe.transform(test_data[NUMERIC_FEATURES])


ordinal_encoder = OrdinalEncoder()
categorical_pipe = Pipeline([
    ('ordinalencoder', ordinal_encoder),
])
categorical_pipe.fit(df[CATEGORICAL_FEATURES])
train_data[CATEGORICAL_FEATURES] = categorical_pipe.transform(train_data[CATEGORICAL_FEATURES])
test_data[CATEGORICAL_FEATURES] = categorical_pipe.transform(test_data[CATEGORICAL_FEATURES])


label_encoder = LabelEncoder()
label_pipe = Pipeline([
    ('labelencoder', ordinal_encoder),
])
label_pipe.fit(df[LABEL].values.reshape(-1, 1))
train_data[LABEL] = label_pipe.transform(train_data[LABEL].values.reshape(-1, 1))
test_data[LABEL] = label_pipe.transform(test_data[LABEL].values.reshape(-1, 1))



train_tensor_X_cat = torch.tensor(train_data[CATEGORICAL_FEATURES].values).long()
# train_tensor_X_cat = train_tensor_X_cat[:, (train_tensor_X_cat != 0).any(axis=0)]
train_tensor_X_num = torch.tensor(train_data[NUMERIC_FEATURES].values).float()
train_tensor_X_num = train_tensor_X_num[:, (train_tensor_X_num != 0).any(axis=0)]
train_tensor_Y = torch.tensor(train_data[LABEL].values).view(-1, 1).float()

test_tensor_X_cat = torch.tensor(test_data[CATEGORICAL_FEATURES].values).long()
# test_tensor_X_cat = test_tensor_X_cat[:, (test_tensor_X_cat != 0).any(axis=0)]
test_tensor_X_num = torch.tensor(test_data[NUMERIC_FEATURES].values).float()
test_tensor_X_num = test_tensor_X_num[:, (test_tensor_X_num != 0).any(axis=0)]
test_tensor_Y = torch.tensor(test_data[LABEL].values).view(-1, 1).float()


# train_tensor_X_cat = torch.randint(0, 5, (3500, 206)) 
# train_tensor_X_num = torch.randn(3500, 53)  
# train_tensor_Y = torch.randn(3500, 1)  

# test_tensor_X_cat = torch.randint(0, 5, (200, 206)) 
# test_tensor_X_num = torch.randn(200, 53)  
# test_tensor_Y = torch.randn(200, 1)


# print("Train Tensor:")
# print(train_tensor_X_cat.shape)
# print(train_tensor_X_num.shape)
# print(train_tensor_Y.shape)

# print("Test Tensor:")
# print(test_tensor_X_cat.shape)
# print(test_tensor_X_num.shape)
# print(test_tensor_Y.shape)


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 100

cat_feature_counts = ()
for column in train_tensor_X_cat.T:
    unique_values = torch.unique(column)
    cat_feature_counts = cat_feature_counts + (len(unique_values),)

num_feature_counts = len(train_tensor_X_num.T)

cont_mean_std = torch.zeros(len(train_tensor_X_num.T), 2)
for i, column in enumerate(train_tensor_X_num.T):
    mean = torch.mean(column)
    std = torch.std(column)
    cont_mean_std[i] = torch.tensor([mean, std])

# print(len(cat_feature_counts), num_feature_counts, cont_mean_std.shape)
# print(cat_feature_counts, num_feature_counts, cont_mean_std)


transformer = FTTransformer(
    categories=cat_feature_counts,
    num_continuous=num_feature_counts,
    dim = 16,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 2,                          # depth, paper recommended 6
    heads = 2,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
)

optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.BCEWithLogitsLoss()
# metrics = AUROC(task="multiclass", num_classes=len(torch.unique(train_tensor_Y)))
metrics = AUROC(task="binary")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.to(device)

# for epoch in tqdm(range(NUM_EPOCHS)):
for epoch in range(NUM_EPOCHS):
    transformer.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = transformer(train_tensor_X_cat, train_tensor_X_num)
    loss = loss_fn(outputs, train_tensor_Y)

    # print("Train")
    # print(train_tensor_X_cat.shape, train_tensor_X_num.shape, train_tensor_Y.shape, loss.shape, outputs.shape)
    # print(train_tensor_X_cat[0:3,:], train_tensor_X_num[0:3,:], train_tensor_Y[0:3,:], outputs[0:3,:])
    # print(outputs)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Evaluation
    transformer.eval()
    with torch.no_grad():
        val_outputs = transformer(test_tensor_X_cat, test_tensor_X_num)
        val_loss = loss_fn(val_outputs, test_tensor_Y)
        # print("Eval")
        # print(test_tensor_X_cat.shape, test_tensor_X_num.shape, test_tensor_Y.shape, val_loss.shape, val_outputs.shape)
        # print(test_tensor_X_cat[0:3,:], test_tensor_X_num[0:3,:], test_tensor_Y[0:3,:], val_outputs[0:3,:])
        val_auc = metrics(val_outputs, test_tensor_Y)
    
    # Print progress
    # if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc.item():.4f}")
