import statistics

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from subject_testing import generate_config_jump3r

results = pd.read_csv('running_time0.csv')
workload_set = pd.read_csv('workload.csv')
config_set, all_combination = generate_config_jump3r()
user_time = results['user_time']
commands = results['command']
replay_mapping = {
        "--replaygain-fast": [1, 0, 0],
        "--replaygain-accurate": [0, 1, 0],
        "--noreplaygain": [0, 0, 1]
    }

ext_set = {
        "wav": [1, 0, 0, 0],
        "mp3": [0, 1, 0, 0],
        "opus": [0, 0, 1, 0],
        "flac": [0, 0, 0, 1]
    }

digital_config = []
for command in commands:
    command_element = command.split(' ')
    numeric_elements = [int(element) for element in command_element if element.isnumeric() or (element.replace(".", "", 1).isdigit() and element.count(".") <= 1)]
    numeric_elements += replay_mapping[command_element[20]]
    file_name = command_element[-2].split('/')[-1]
    if file_name == ".DS_Store":
        digital_config.append([])
        continue
    workload_data = workload_set[workload_set['file_name'].str.contains(file_name)].values.tolist()[0]
    numeric_elements += workload_data[0:3]
    numeric_elements += ext_set[workload_data[3]]
    digital_config.append(numeric_elements)

user_time_list = []
for i, element in enumerate(user_time):
    if element.endswith("user"):
        element = float(element.replace('user', ''))
        config = digital_config[i]
        config += [element]
        user_time_list.append(config)
df_merge = pd.DataFrame(user_time_list)
df_merge = df_merge.fillna(0)
df_merge = df_merge.loc[:, (df_merge != 0).any()]

scaler = MinMaxScaler()
# Fit the scaler to your data and transform it
scaled_data = scaler.fit_transform(df_merge)
# Create a new DataFrame with the scaled values
scaled_df_merge = pd.DataFrame(scaled_data, columns=df_merge.columns)

X = scaled_df_merge.iloc[:, :-1]  # Features (all columns except the last one)
y = scaled_df_merge.iloc[:, -1]   # Target variable (last column)

train_size = X.shape[1]
train_ratio = float(train_size/X.shape[0])
# Step 4: Split the data into training and testing sets

for n in range(1, 7):
    error_set = []
    for k in range(1, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-n*train_ratio), random_state=42)

        # Step 5: Train an XGBoost model
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the Variance of the actual target values
        variance_y = np.var(y_test)

        # Calculate the Relative Squared Error (RSE) for each prediction
        rse = ((y_test - y_pred) ** 2) / variance_y

        # Calculate the Median RSE
        mrse = np.median(rse)
        #print("Relative Mean Squared Error: {:.4f}".format(np.mean(rse)))
        error_set.append(np.mean(rse))
    print(statistics.mean(error_set))
