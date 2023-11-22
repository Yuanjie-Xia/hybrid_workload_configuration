import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from subject_testing import generate_config_jump3r
from method_implement.mtad_gat import mtad_impl


def load_results(results):
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
    return df_merge


def preprocessing(df_merge):
    scaler = MinMaxScaler()
    # Fit the scaler to your data and transform it
    scaled_data = scaler.fit_transform(df_merge)
    # Create a new DataFrame with the scaled values
    scaled_df_merge = pd.DataFrame(scaled_data, columns=df_merge.columns)
    scaled_df_merge = scaled_df_merge.iloc[:, :-1]
    # Merge df1_dropped with df2
    scaled_df_merge = pd.concat([scaled_df_merge, df_merge.iloc[:, -1]], axis=1)
    return scaled_df_merge


def relative_error(y_pred, y_test):
    errors = []
    for pred, actual in zip(y_pred, y_test):
        if actual != 0:
            error = abs(pred - actual) / abs(actual)
            errors.append(error)
        else:
            # Handle the case where the actual value is zero to avoid division by zero
            errors.append(float('inf'))  # Assigning infinity for the relative error
    return errors


def median_relative_error(predictions, targets):
    absolute_errors = torch.abs(predictions - targets)
    relative_errors = absolute_errors / (torch.abs(targets) + 1e-8)
    return torch.median(relative_errors).item()


def main():
    for num in range(0, 3):
        print(num)
        results = pd.read_csv('jump3r_result/running_time' + str(num) + '.csv')
        average_user_time = load_results(results)
        print(average_user_time.shape)
        scaled_df = preprocessing(average_user_time)
        # simply_merge(scaled_df, num)
        mtad_impl(scaled_df, num)


if __name__ == "__main__":
    main()