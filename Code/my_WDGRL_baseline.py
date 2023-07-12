import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn import metrics
from adapt.utils import make_classification_da
from adapt.feature_based import WDGRL

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

print("packages loaded")

seq_featureVec = np.load("./all_onehotendcoded_seq.npy",
                         allow_pickle=True)

print("length of one hot encodings", len(seq_featureVec))
length_checker = np.vectorize(len)
arr_len = length_checker(seq_featureVec)
X_padded = []
for i in range(len(seq_featureVec)):
    tmp = padarray(seq_featureVec[i], max(arr_len))
    X_padded.append(tmp)
print("max len", max(arr_len))
print("len of X_padded", len(X_padded))
X = X_padded

variant_data = np.load("./all_variantNames.npy")
print("length of variants", len(variant_data))

#######################################################
#      Create dictionary for class indexes
#######################################################
idx_to_class = {i:j for i, j in enumerate(np.unique(variant_data))}
class_to_idx = {value:key for key,value in idx_to_class.items()}

y = []
for i in range(len(variant_data)):
    y.append(class_to_idx[variant_data[i]])
print("length of y ", len(y))

####################################################
#       Create Train, Valid and Test sets
####################################################
# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2,
                                                  stratify=y_trainval, random_state=21)

print("len X_train", len(X_train))
print("len y_train", len(y_train))
print("len X_test", len(X_test))
print("len y_test", len(y_test))
print("len X_val", len(X_val))
print("len y_val", len(y_val))

####################################################
#       normalize data to [0,1] range
####################################################
X_train, y_train = np.array(X_train,dtype=float), np.array(y_train,dtype=int)
X_val, y_val = np.array(X_val,dtype=float), np.array(y_val,dtype=int)
X_test, y_test = np.array(X_test,dtype=float), np.array(y_test,dtype=int)

####################################################
#       transform inputs to low dim  using WDGRL
####################################################
model = WDGRL(lambda_=1., gamma=1., Xt=X_test, metrics=["acc"], random_state=0)
clf = model.fit(X_train, y_train, epochs=100, verbose=0)
y_pred = clf.predict(X_test)
print(model.score(X_test, y_test))
X_train = model.transform(X_train)
X_test = model.transform(X_test)
X_val = model.transform(X_val)

np.save("./x_train_wdgrl_var.npy", X_train)
np.save("./x_test_wdgrl_var.npy", X_test)
np.save("./x_val_wdgrl_var.npy", X_val)
np.save("./y_train_wdgrl_var.npy", y_train)
np.save("./y_test_wdgrl_var.npy", y_test)
np.save("./y_val_wdgrl_var.npy", y_val)
