import numpy as np
import pandas as pd
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(script_dir, 'sample_data')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

while True:
    sample_choice = int(input("Choose Normal(0) or Hypertension(1): "))

    if sample_choice == 0:
        original_file_path = os.path.join(sample_data_dir, 'sample_normal.csv')
        scaled_file_path = os.path.join(sample_data_dir, 'sample_normal_scaled.csv')

        original_data = pd.read_csv(original_file_path)
        scaled_data = pd.read_csv(scaled_file_path)
        break

    elif sample_choice == 1:
        original_file_path = os.path.join(sample_data_dir, 'sample_hypertension.csv')
        scaled_file_path = os.path.join(sample_data_dir, 'sample_hypertension_scaled.csv')

        original_data = pd.read_csv(original_file_path)
        scaled_data = pd.read_csv(scaled_file_path)
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
print(original_data.iloc[sample_number])
print("--------------------------")

prob = model.predict_proba([scaled_data.iloc[sample_number]])
prob = prob[:,1]

prob2 = round(prob[0]*100, 2)

if prob >= 0.4751480317013699:
    print("Hypertension, Probability : {}%".format(prob2))

else:
    print("Normal, Probability : {}%".format(prob2))