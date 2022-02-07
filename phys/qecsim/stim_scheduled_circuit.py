from math import exp
from typing import Tuple, Callable

import stim

from qecsim.qec_generator import CircuitParams


def _get_pauli_probs(duration: float, t1: float, t2: float) -> Tuple[float, float, float]:
    """
    calculate the Pauli depolarization probabilities
    according to the twirl approximation in eq. 10 in https://arxiv.org/abs/1210.5799
    :param duration: duration of the instruction
    :param t1:
    :param t2:
    :return: (p_x, p_y, p_z)
    """
    return (1 - exp(-duration / t1)) / 4, (1 - exp(-duration / t1)) / 4, (1 - exp(-duration / t2)) / 2 - (
            1 - exp(-duration / t1)) / 4


def _updater(circ: stim.Circuit, inst_handler: Callable):
    new_circ = stim.Circuit()
    for inst in circ:
        inst: stim.CircuitInstruction
        if isinstance(inst, stim.CircuitRepeatBlock):
            new_circ += _updater(inst.body_copy(), inst_handler) * inst.repeat_count
        else:
            inst_handler(inst, new_circ)
    return new_circ


def generate_scheduled(code_task: str,
                       distance: int,
                       rounds: int,
                       params: CircuitParams) -> stim.Circuit:
    """

    :param code_task:
    :param distance:
    :param rounds:
    :param params:
    :return:
    """
    circuit = stim.Circuit.generated(code_task,
                                     distance=distance,
                                     rounds=rounds,
                                     after_clifford_depolarization=params.two_qubit_depolarization_rate,
                                     before_measure_flip_probability=0,
                                     after_reset_flip_probability=0
                                     )
    qubit_indices = {inst.targets_copy()[0].value for inst in circuit if
                     isinstance(inst, stim.CircuitInstruction) and inst.name == 'QUBIT_COORDS'}

    inst_with_duration = {'CX', 'H', 'MR', 'M'}

    def instruction_duration(inst: stim.CircuitInstruction) -> float:
        if inst.name not in inst_with_duration:
            ValueError(f'only duration of {inst_with_duration} is known')
        if inst.name == 'H':
            return params.single_qubit_gate_duration
        if inst.name == 'CX':
            return params.two_qubit_gate_duration
        if inst.name == 'MR':
            return params.reset_duration + params.reset_latency + params.meas_duration
        if inst.name == 'M':
            return params.meas_duration

    def add_t1_t2_depolarization(inst: stim.CircuitInstruction, new_circ: stim.Circuit) -> None:
        if inst.name in inst_with_duration:
            idle_qubits = qubit_indices.difference(t.value for t in inst.targets_copy())
            new_circ.append_operation(inst)
            new_circ.append_operation('PAULI_CHANNEL_1',
                                      list(idle_qubits),
                                      _get_pauli_probs(instruction_duration(inst),
                                                       params.t1,
                                                       params.t2
                                                       )
                                      )
        else:
            new_circ.append_operation(inst)

    def replace_mr_with_m(inst: stim.CircuitInstruction, new_circ: stim.Circuit) -> None:
        if inst.name == 'MR':
            new_circ.append_operation('M', inst.targets_copy(), inst.gate_args_copy())
        else:
            new_circ.append_operation(inst)

    def separate_depolarization(inst: stim.CircuitInstruction, new_circ: stim.Circuit) -> None:
        if inst.name == 'DEPOLARIZE1':
            new_circ.append_operation('DEPOLARIZE1',
                                      inst.targets_copy(),
                                      params.single_qubit_depolarization_rate)
        elif inst.name == 'DEPOLARIZE2':
            new_circ.append_operation('DEPOLARIZE2',
                                      inst.targets_copy(),
                                      params.two_qubit_depolarization_rate)
        else:
            new_circ.append_operation(inst)

    circuit = _updater(circuit, add_t1_t2_depolarization)
    circuit = _updater(circuit, separate_depolarization)
    circuit = _updater(circuit, replace_mr_with_m)
    return circuit


if __name__ == '__main__':
    cparams = CircuitParams(t1=15e3,
                            t2=19e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=100,
                            single_qubit_depolarization_rate=1.1e-3,
                            two_qubit_depolarization_rate=6.6e-3,
                            meas_duration=600,
                            reset_duration=0,
                            reset_latency=40)
    # print(reversed(genc))
    print(generate_scheduled('surface_code:rotated_memory_z',
                             distance=3,
                             rounds=4,
                             params=cparams))

#        print(type(inst), '\t', inst)
