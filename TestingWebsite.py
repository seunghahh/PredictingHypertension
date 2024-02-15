import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay, auc

with open('adb_lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

while True:
    sample_choice = int(input("Choose normal(0) or hypertension(1): "))

    if sample_choice == 0:
        original_data = pd.read_csv('sample_normal.csv')
        scaled_data = pd.read_csv('sample_normal_scaled.csv')
        break

    elif sample_choice == 1:
        original_data = pd.read_csv('sample_hypertension.csv')
        scaled_data = pd.read_csv('sample_hypertension_scaled.csv')
        break

    else:
        print("Type 0 or 1")

sample_number = int(input("Choose the number 1-100: "))
sample_number = sample_number - 1

print("Data of Patent")
print(original_data.iloc[sample_number])

prob = model.predict_proba([scaled_data.iloc[sample_number]])
prob = prob[:,1]

prob2 = round(prob[0]*100, 2)

if prob >= 0.4751480317013699:
    print("Hypertension, probability : {}%".format(prob2))

else:
    print("Normal, probability : {}%".format(prob2))