from importlib import import_module


def get_nets(net_name_list):
    Net_dict = {}
    for net_name in net_name_list:
        Net = import_module("nets." + net_name).Net
        Net_dict[net_name] = Net
    return Net_dict
