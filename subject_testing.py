from itertools import chain, combinations


def powerset(list_name):
    s = list(list_name)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_config_jump3r():
    config_option = []
    all_possible_configs = []
    configs = []
    all_combination = powerset(configs)
    for x in all_combination:
        if len(x) > 0:
            cmd = ''
            for item in x:
                cmd += item
            config_option.append(cmd)
    for a1 in range(0, 2):
        for a2 in range(0, 2):
            for a3 in range(0, 2):
                for a4 in range(0, 2):
                    for a5 in range(0, 2):
                        for a6 in range(0, 2):
                            for a7 in range(0, 2):
                                for a8 in range(0, 2):
                                    for a9 in range(0, 2):
                                        all_possible_configs.append([a1, a2, a3, a4, a5, a6, a7, a8, a9])
    all_possible_configs.remove([0]*9)
    return config_option, all_possible_configs