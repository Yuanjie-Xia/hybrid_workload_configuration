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


def generate_config_x264():
    asm = ['', '--no-asm ']
    x8dct = ['', '--no-8x8dct ']
    cabac = ['', '--no-cabac ']
    deblock = ['', '--no-deblock ']
    pskip = ['', '--no-fast-pskip ']
    mbtree = ['', '--no-mbtree ']
    mixed_refs = ['', '--no-mixed-refs ']
    weightb = ['', '--no-weightb ']
    rc_lookahead = ['--rc-lookahead 20 ', '--rc-lookahead 40 ']
    ref = ['--ref 1 ', '--ref 5 ', '--ref 9 ']
    # Create a list of all your lists
    lists = [asm, x8dct, cabac, deblock, pskip, mbtree, mixed_refs, weightb, rc_lookahead, ref]
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
                command = "../x264/x264 " + comb + " -o " + file + " output.mp3"
                command_list.append(command)
    return command_list


if __name__ == "__main__":
    combinations, _ = generate_command_dconvert()
    command_list = generate_command_dconvert(combinations, "./music_encorder_test_set")
