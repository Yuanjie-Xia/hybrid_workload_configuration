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


def generate_config_dconvert():
    platform = ['-androidMipmapInsteadOfDrawable ', '-androidIncludeLdpiTvdpi ', '-iosCreateImagesetFolders ']
    compression = ['-low_compression ', '-high_compression ']
    thread = ['-threads 1 ', '-threads 2 ', '-threads 3 ']
    scale = ['-scale 10 ', '-scale 90 ']
    # Create a list of all your lists
    lists = [platform, compression, thread, scale]
    # Get all combinations using itertools.product
    all_combinations = list(product(*lists))

    # Merge the combinations into one string with " " as a delimiter
    merged_combinations = [" ".join(map(str, combo)) for combo in all_combinations]
    return merged_combinations, all_combinations


def generate_command_dconvert(combinations, root_dir):
    command_list = []
    # Example usage:
    all_files = find_files(root_dir)
    for file in all_files:
        if ~file.startswith('.'):
            for comb in combinations:
                command = ".java -jar ./dconvert.jar -src " + file + " -o " + file + " output.mp3"
                command_list.append(command)
    return command_list


if __name__ == "__main__":
    combinations, _ = generate_command_dconvert()
    command_list = generate_command_dconvert(combinations, "./music_encorder_test_set")
