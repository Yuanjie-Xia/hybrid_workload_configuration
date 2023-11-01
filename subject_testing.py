from itertools import chain, combinations, product


def powerset(list_name):
    s = list(list_name)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_config_jump3r():
    config_option = []
    all_possible_configs = []
    configs = []
    lowpass = ["--lowpass 5000", "--lowpass 10000", "--lowpass 20000"]
    lowpass_width = ["", "--lowpass-width 1000", "--lowpass-width 2000"]
    highpass = ["--highpass 20000", "--highpass 25000", "--highpass 30000"]
    highpass_wide = ["", "--highpass-width 1000", "--highpass-width 2000"]
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
    # Print the combinations
    for combo in merged_combinations:
        print(combo)
        break
    print(len(merged_combinations))
    return config_option, all_possible_configs


if __name__ == "__main__":
    generate_config_jump3r()