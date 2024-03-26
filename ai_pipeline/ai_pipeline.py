import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.trial import TrialState
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
from tab_transformer_pytorch import TabTransformer, FTTransformer
from torchmetrics import AUROC
from tqdm import tqdm
import sys
import torch.optim as optim

###############################

transformer_type = sys.argv[1]
assert transformer_type in ["tabt", "tft"], "Invalid transformer_type. Must be 'tabt' or 'tft'."

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



# train_tensor_X_cat = torch.tensor(train_data[CATEGORICAL_FEATURES].values).long()
# # train_tensor_X_cat = train_tensor_X_cat[:, (train_tensor_X_cat != 0).any(axis=0)]
# train_tensor_X_num = torch.tensor(train_data[NUMERIC_FEATURES].values).float()
# train_tensor_X_num = train_tensor_X_num[:, (train_tensor_X_num != 0).any(axis=0)]
# train_tensor_Y = torch.tensor(train_data[LABEL].values).view(-1, 1).float()

# test_tensor_X_cat = torch.tensor(test_data[CATEGORICAL_FEATURES].values).long()
# # test_tensor_X_cat = test_tensor_X_cat[:, (test_tensor_X_cat != 0).any(axis=0)]
# test_tensor_X_num = torch.tensor(test_data[NUMERIC_FEATURES].values).float()
# test_tensor_X_num = test_tensor_X_num[:, (test_tensor_X_num != 0).any(axis=0)]
# test_tensor_Y = torch.tensor(test_data[LABEL].values).view(-1, 1).float()


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



# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func

#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss

def objective(trial):
    ### Get data
    CV_train_data, CV_val_data = train_test_split(train_data, stratify=train_data[LABEL], test_size=0.10, random_state=seed)
    
    ### 
    CV_train_tensor_X_cat = torch.tensor(CV_train_data[CATEGORICAL_FEATURES].values).long()
    CV_train_tensor_X_num = torch.tensor(CV_train_data[NUMERIC_FEATURES].values).float()
    CV_train_tensor_X_num = CV_train_tensor_X_num[:, (CV_train_tensor_X_num != 0).any(axis=0)]
    CV_train_tensor_Y = torch.tensor(CV_train_data[LABEL].values).view(-1, 1).float()

    CV_val_tensor_X_cat = torch.tensor(CV_val_data[CATEGORICAL_FEATURES].values).long()
    CV_val_tensor_X_num = torch.tensor(CV_val_data[NUMERIC_FEATURES].values).float()
    CV_val_tensor_X_num = CV_val_tensor_X_num[:, (CV_val_tensor_X_num != 0).any(axis=0)]
    CV_val_tensor_Y = torch.tensor(CV_val_data[LABEL].values).view(-1, 1).float()

    # print("Train Tensor:")
    # print(CV_train_tensor_X_cat.shape)
    # print(CV_train_tensor_X_num.shape)
    # print(CV_train_tensor_Y.shape)

    # print("Test Tensor:")
    # print(CV_val_tensor_X_cat.shape)
    # print(CV_val_tensor_X_num.shape)
    # print(CV_val_tensor_Y.shape)

    ### Counts of features
    cat_feature_counts = ()
    for column in CV_train_tensor_X_cat.T:
        unique_values = torch.unique(column)
        cat_feature_counts = cat_feature_counts + (len(unique_values),)

    num_feature_counts = len(CV_train_tensor_X_num.T)

    cont_mean_std = torch.zeros(len(CV_train_tensor_X_num.T), 2)
    for i, column in enumerate(CV_train_tensor_X_num.T):
        mean = torch.mean(column)
        std = torch.std(column)
        cont_mean_std[i] = torch.tensor([mean, std])

    ### TabTransformer Hyperparameters
    learning_rate_hyp = 0.001
    embedding_dim_hyp = trial.suggest_categorical('embedding_dim',[16])
    depth_hyp = trial.suggest_categorical('depth', [1])
    heads_hyp = trial.suggest_categorical('heads', [2])
    attn_dropout_hyp = 0.01
    ff_dropout_hyp = 0.01
    # learning_rate_hyp = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    # embedding_dim_hyp = trial.suggest_categorical('embedding_dim',[8, 16, 32, 64])
    # depth_hyp = trial.suggest_categorical('depth', [1, 2, 3, 4])
    # heads_hyp = trial.suggest_categorical('heads', [2, 3, 4, 6])
    # attn_dropout_hyp = trial.suggest_float("attn_dropout", 0.01, 0.5)
    # ff_dropout_hyp = trial.suggest_float("ff_dropout", 0.01, 0.5)
    # mlp_hidden_factor1_hyp = trial.suggest_categorical("mlp_hidden_factor1", [0.5, 1, 2])
    # mlp_hidden_factor2_hyp = trial.suggest_categorical("mlp_hidden_factor2", [0.5, 1, 2])

    ### TabTransformer Model
    if transformer_type == "tabt":
        transformer = TabTransformer(
            categories=cat_feature_counts,
            num_continuous=num_feature_counts,
            dim=embedding_dim_hyp,
            dim_out=1,
            depth=depth_hyp,
            heads=heads_hyp,
            attn_dropout=attn_dropout_hyp,
            ff_dropout=ff_dropout_hyp,
            # mlp_hidden_mults=(mlp_hidden_factor1_hyp, mlp_hidden_factor2_hyp),
            mlp_act=torch.nn.ReLU(),
            continuous_mean_std=cont_mean_std
        )
    elif transformer_type == "tft":
        transformer = FTTransformer(
            categories=cat_feature_counts,
            num_continuous=num_feature_counts,
            dim = embedding_dim_hyp,
            dim_out = 1,
            depth = depth_hyp,
            heads = heads_hyp,
            attn_dropout = attn_dropout_hyp,
            ff_dropout = ff_dropout_hyp
        )
    else:
        raise ValueError("Invalid transformer_type. Must be 'tabt' or 'tft'.")

    LEARNING_RATE = learning_rate_hyp
    NUM_EPOCHS = 100

    optimizer = optim.AdamW(
            transformer.parameters(),
            lr=LEARNING_RATE
        )
    
    loss_fn = nn.BCEWithLogitsLoss()

    # metrics = AUROC(task="multiclass", num_classes=len(torch.unique(train_tensor_Y)))
    metrics = AUROC(task="binary")
    
    # early = EarlyStopping(patience=20, verbose=False)
    # callback_list = [early]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)

    for epoch in tqdm(range(NUM_EPOCHS)):
        transformer.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = transformer(CV_train_tensor_X_cat, CV_train_tensor_X_num)
        loss = loss_fn(outputs, CV_train_tensor_Y)

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
            val_outputs = transformer(CV_val_tensor_X_cat, CV_val_tensor_X_num)
            val_loss = loss_fn(val_outputs, CV_val_tensor_Y)
            # print("Eval")
            # print(test_tensor_X_cat.shape, test_tensor_X_num.shape, test_tensor_Y.shape, val_loss.shape, val_outputs.shape)
            # print(test_tensor_X_cat[0:3,:], test_tensor_X_num[0:3,:], test_tensor_Y[0:3,:], val_outputs[0:3,:])
            val_auc = metrics(val_outputs, CV_val_tensor_Y)

        trial.report(val_auc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # Print progress
        # if (epoch + 1) % 10 == 0:
        # print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc.item():.4f}")

    gc.collect()
    
    return val_auc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# test_tensor_X_cat = torch.tensor(test_data[CATEGORICAL_FEATURES].values).long()
# test_tensor_X_num = torch.tensor(test_data[NUMERIC_FEATURES].values).float()
# test_tensor_X_num = test_tensor_X_num[:, (test_tensor_X_num != 0).any(axis=0)]
# test_tensor_Y = torch.tensor(test_data[LABEL].values).view(-1, 1).float()

































# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 0.0001
# NUM_EPOCHS = 100

# cat_feature_counts = ()
# for column in train_tensor_X_cat.T:
#     unique_values = torch.unique(column)
#     cat_feature_counts = cat_feature_counts + (len(unique_values),)

# num_feature_counts = len(train_tensor_X_num.T)

# cont_mean_std = torch.zeros(len(train_tensor_X_num.T), 2)
# for i, column in enumerate(train_tensor_X_num.T):
#     mean = torch.mean(column)
#     std = torch.std(column)
#     cont_mean_std[i] = torch.tensor([mean, std])

# print(len(cat_feature_counts), num_feature_counts, cont_mean_std.shape)
# print(cat_feature_counts, num_feature_counts, cont_mean_std)


# transformer = TabTransformer(
#     categories=cat_feature_counts,
#     num_continuous=num_feature_counts,
#     dim=16,
#     dim_out=1,
#     depth=2,
#     heads=2,
#     attn_dropout=0.1,
#     ff_dropout=0.1,
#     mlp_hidden_mults=(4, 2),
#     mlp_act=nn.ReLU(),
#     continuous_mean_std=cont_mean_std
# )

# optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# loss_fn = nn.BCEWithLogitsLoss()
# # metrics = AUROC(task="multiclass", num_classes=len(torch.unique(train_tensor_Y)))
# metrics = AUROC(task="binary")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transformer.to(device)

# for epoch in tqdm(range(NUM_EPOCHS)):
#     transformer.train()
#     optimizer.zero_grad()
    
#     # Forward pass
#     outputs = transformer(train_tensor_X_cat, train_tensor_X_num)
#     loss = loss_fn(outputs, train_tensor_Y)

#     # print("Train")
#     # print(train_tensor_X_cat.shape, train_tensor_X_num.shape, train_tensor_Y.shape, loss.shape, outputs.shape)
#     # print(train_tensor_X_cat[0:3,:], train_tensor_X_num[0:3,:], train_tensor_Y[0:3,:], outputs[0:3,:])
#     print(outputs)

#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()
    
#     # Evaluation
#     transformer.eval()
#     with torch.no_grad():
#         val_outputs = transformer(test_tensor_X_cat, test_tensor_X_num)
#         val_loss = loss_fn(val_outputs, test_tensor_Y)
#         # print("Eval")
#         # print(test_tensor_X_cat.shape, test_tensor_X_num.shape, test_tensor_Y.shape, val_loss.shape, val_outputs.shape)
#         # print(test_tensor_X_cat[0:3,:], test_tensor_X_num[0:3,:], test_tensor_Y[0:3,:], val_outputs[0:3,:])
#         val_auc = metrics(val_outputs, test_tensor_Y)
    
#     # Print progress
#     # if (epoch + 1) % 10 == 0:
#     print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc.item():.4f}")
