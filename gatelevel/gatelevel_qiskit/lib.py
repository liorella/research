import importlib
from copy import deepcopy

from matplotlib import pyplot as plt
from qiskit.pulse import ShiftPhase, Play, Acquire


def write_indent_line(st, level=0):
    return " " * level * 4 + st + "\n"


def wfs_no_samples(wfs: dict):
    wfs_new = deepcopy(wfs)
    for key in wfs_new.keys():
        for element in wfs_new[key]:
            element.pop('samples')
    return wfs_new


def summary_of_inst(inst_seq):
    gen_list = []
    for inst in inst_seq:
        if isinstance(inst[1], ShiftPhase):
            gen_list.append(inst)
        elif isinstance(inst[1], Play):
            gen_list.append((inst[0], inst[1].name, inst[1].channel))
        elif isinstance(inst[1], Acquire):
            gen_list.append(inst)
        else:

            raise ValueError(f"unknown instruction {inst}")
    return gen_list


def plot_waveforms(play_inst):
    plt.plot(play_inst[1].pulse.samples.real)
    plt.plot(play_inst[1].pulse.samples.imag)
    plt.legend(('real', 'imaginary'))
    plt.title(play_inst[1].name)


def get_min_time(wfs_data):
    mins = []
    for key in wfs_data.keys():
        mins.append(min(element['timestamp'] for element in wfs_data[key]))
    return min(mins)
