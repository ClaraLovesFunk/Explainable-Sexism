import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from EDA import *
from experts_modules import *

results = np.load(f'results_master_bb.npy',allow_pickle='TRUE').item()
print('-----------------------------------------------------------')
print('-----------------------------------------------------------')
print(f'f1-macro_avrg: {results["f1-macro_avrg"]}')
print(f'f1-no_avrg: {results["f1-no_avrg"]}')
print(f'acc: {results["acc"]}')


results = np.load(f'results_master_hh.npy',allow_pickle='TRUE').item()
print('-----------------------------------------------------------')
print('-----------------------------------------------------------')
print(f'f1-macro_avrg: {results["f1-macro_avrg"]}')
print(f'f1-no_avrg: {results["f1-no_avrg"]}')
print(f'acc: {results["acc"]}')