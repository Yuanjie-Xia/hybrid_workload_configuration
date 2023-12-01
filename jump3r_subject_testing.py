from itertools import chain, combinations, product
import os


def find_files(root_dir):
    file_paths = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(foldername, filename))
    return file_paths


def powerset(list_name):
    s = list(list_name)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_config_jump3r():
    lowpass = ["--lowpass 5000", "--lowpass 10000", "--lowpass 20000"]
    lowpass_width = ["--lowpass-width 1000", "--lowpass-width 2000"]
    highpass = ["--highpass 20000", "--highpass 25000", "--highpass 30000"]
    highpass_wide = ["--highpass-width 1000", "--highpass-width 2000"]
    cbr = ["-b 8", "-b 160", "-b 320"]
    abr = ["--abr 8", "--abr 160", "--abr 320"]
    vbr = ["-V 0", "-V 4", "-V 9"]
    resample = ["--resample 8", "--resample 22", "--resample 44", "--resample 48"]
    replay = ["--replaygain-fast", "--replaygain-accurate", "--noreplaygain"]
    # Create a list of all your lists
    lists = [lowpass, lowpass_width, highpass, highpass_wide, cbr, abr, vbr, resample, replay]
    # Get all combinations using itertools.product
    all_combinations = list(product(*lists))

    # Merge the combinations into one string with " " as a delimiter
    merged_combinations = [" ".join(map(str, combo)) for combo in all_combinations]
    return merged_combinations, all_combinations


def generate_command_jump3r(combinations, root_dir):
    command_list = []
    # Example usage:
    all_files = find_files(root_dir)
    for file in all_files:
        for comb in combinations:
            command = "time java -jar jump3r.jar " + comb + " " + file + " output.mp3"
            command_list.append(command)
    return command_list


if __name__ == "__main__":
    combinations, _ = generate_config_jump3r()
    command_list = generate_command_jump3r(combinations, "./music_encorder_test_set")
