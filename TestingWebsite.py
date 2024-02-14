import numpy as np
import pandas as pd
import pickle
import warnings
import random
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay, auc

with open('adb_lr_model.pkl', 'rb') as f:
    model = pickle.load(f)