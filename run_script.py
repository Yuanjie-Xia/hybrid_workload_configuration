import pandas as pd
import os
import subprocess
import time
import datetime
import tqdm
import concurrent.futures  # Add this import

import sys
from subject_testing import generate_config_jump3r, generate_command_jump3r


def execute_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[1].decode("utf-8")
    perf_info = p.split(" ")
    user_time = perf_info[0]
    system = perf_info[1]
    elapsed = perf_info[2]
    cpu = perf_info[3]
    IO = perf_info[5]
    pagefaults = perf_info[6]
    swaps = perf_info[7]
    return [command, user_time, system, elapsed, cpu, IO, pagefaults, swaps, i]


def main():
    combinations = generate_config_jump3r()
    command_list = generate_command_jump3r(combinations, "./music_encorder_test_set")

    num_threads = 4  # Specify the number of threads you want to use

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(execute_command, command_list))

    for i in range(10):
        running_time_df = pd.DataFrame(results, columns=["command", "user_time", "system", "elapsed", "cpu", "IO", "pagefaults", "swaps", "times"])
        running_time_df.to_csv("running_time" + str(i) + ".csv")


if __name__ == "__main__":
    main()