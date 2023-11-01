import pandas as pd
import os
import subprocess
import time
import datetime

import sys
from subject_testing import generate_config_jump3r, generate_command_jump3r


def main():
    combinations = generate_config_jump3r()
    command_list = generate_command_jump3r(combinations, "./music_encorder_test_set")

    for i in range(10):
        df = []
        # directory = os.getcwd()
        # print(directory)
        for command in command_list:
            p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE).communicate()[0]
            print(p)
            perf_info = p.split("\n")[-1]
            perf_info_element = perf_info.split(" ")
            operation_time = perf_info_element[-8]
            user_time = perf_info_element[-6]
            system = perf_info_element[-4]
            cpu = perf_info_element[-2]
            df.append([command, operation_time, user_time, system, cpu])

        running_time_df = pd.DataFrame(df, columns=["command", "operation_time", "user_time", "system", "cpu"])
        running_time_df.to_csv("running_time" + str(i) + ".csv")


if __name__ == "__main__":
    main()
