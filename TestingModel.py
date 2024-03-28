import numpy as np
import pandas as pd
import pickle
import os
import scipy.io as sio

def load_mat_data(file_name):
    contentsMat = sio.loadmat(file_name)
    arr_min = np.squeeze(contentsMat['arr_min'])
    arr_max = np.squeeze(contentsMat['arr_max'])
    return arr_max, arr_min

def min_max_test_datasets(test_data, input_factor, arr_min, arr_max):    ###=== Copy Input Data
   
    ###=== Copy Input Data
    edit_data = test_data.copy()
    edit_data = np.reshape(edit_data, (1, input_factor))
 
    ###=== Find Nan Values and Fill-up Nan Values
    for idx_col in range(0, input_factor):
        edit_data[np.isnan(edit_data[:, idx_col]), idx_col] = arr_min[idx_col]

    ###=== Normalization Values
    for idx_col in range(0, input_factor):
        max_data = arr_max[idx_col]
        min_data = arr_min[idx_col]
        edit_data[:, idx_col] = (edit_data[:, idx_col] - min_data) / (max_data - min_data)
    return edit_data

def eval_model_ML(input_data, input_factor = 18):
    ### Scaling Datasets
    arr_max, arr_min = load_mat_data(script_dir + '/hypertension_norm.mat')
    input_data = min_max_test_datasets(input_data.copy(), input_factor, arr_min, arr_max)

    ### Evaluation Model with DNN
    print('Evaluation with Testing')
    #model = pickle.load(open(SavePath+'adb_lr_model.pkl', 'rb'))
    with open(model_dir, 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict_proba(input_data)[:,1]#.squeeze()
    return prediction

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(script_dir, 'sample_data')
model_dir = os.path.join(script_dir, 'model.pkl')

while True:
    sample_choice = int(input("Choose Normal(0) or Hypertension(1): "))

    if sample_choice == 0:
        file_path = os.path.join(sample_data_dir, 'sample_normal.csv')
        data = pd.read_csv(file_path)
        break

    elif sample_choice == 1:
        file_path = os.path.join(sample_data_dir, 'sample_hypertension.csv')
        data = pd.read_csv(file_path)
        break

    else:
        print("Type 0 or 1")

while True:
    sample_number = int(input("Choose the number 1-100: "))

    if (sample_number >= 1) and (sample_number <= 100):
        sample_number = sample_number - 1
        break

    else:
        print("Please enter a number between 1 and 100")

print("Data of Patent")
print("--------------------------")
print(data.iloc[sample_number])
print("--------------------------")

input_data = np.array(data.iloc[[sample_number]])

vProb = eval_model_ML(input_data)

vProb2 = round(vProb[0]*100, 2)

if vProb2 >= 0.5:
    print("Hypertension, Probability : {}%".format(vProb2))

else:
    print("Normal, Probability : {}%".format(vProb2))