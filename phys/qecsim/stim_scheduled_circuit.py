from itertools import takewhile
from math import exp
from typing import Tuple, Callable
import numpy.typing as npt

import numpy as np
import stim

from qecsim.qec_generator import CircuitParams


def get_pauli_probs(duration: float, t1: float, t2: float) -> Tuple[float, float, float]:
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


def to_scheduled_circuit(circuit: stim.Circuit, params: CircuitParams) -> stim.Circuit:
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
                                      get_pauli_probs(instruction_duration(inst),
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
            new_circ.append_operation('DEPOLARIZE1', inst.targets_copy(), params.single_qubit_depolarization_rate)
        elif inst.name == 'DEPOLARIZE2':
            new_circ.append_operation('DEPOLARIZE2', inst.targets_copy(), params.two_qubit_depolarization_rate)
        else:
            new_circ.append_operation(inst)

    def updater(circ: stim.Circuit, inst_handler: Callable):
        new_circ = stim.Circuit()
        for inst in circ:
            inst: stim.CircuitInstruction
            if isinstance(inst, stim.CircuitRepeatBlock):
                new_circ += updater(inst.body_copy(), inst_handler) * inst.repeat_count
            else:
                inst_handler(inst, new_circ)
        return new_circ

    circuit = updater(circuit, add_t1_t2_depolarization)
    circuit = updater(circuit, separate_depolarization)
    circuit = updater(circuit, replace_mr_with_m)
    return circuit


def get_matching_matrix(circuit: stim.Circuit) -> np.ndarray:
    def get_records(circ: stim.Circuit):
        m = []
        for inst in circ:
            if isinstance(inst, stim.CircuitRepeatBlock):
                for _ in range(inst.repeat_count):
                    m.extend(get_records(inst.body_copy()))
            elif inst.name in ('M', 'MR'):
                m.extend(t.value for t in inst.targets_copy())
        return m

    measures = get_records(circuit)
    match_indices = []
    last_detectors = takewhile(lambda x: x.name == 'DETECTOR', reversed(circuit[:-1]))
    for detector in last_detectors:
        match_indices.append([measures[k] for k in [t.value for t in detector.targets_copy()]])
    ancillas = sorted(r[-1] for r in match_indices)
    data_qubits = []
    for r in match_indices:
        data_qubits.extend(r[:-1])
    data_qubits = sorted(set(data_qubits))
    match_matrix = np.zeros((len(ancillas), len(data_qubits)), dtype=np.uint8)
    for r in match_indices:
        match_matrix[ancillas.index(r[-1]), [data_qubits.index(i) for i in r[:-1]]] = 1
    return match_matrix
    # todo: create a matching object that doesn't have duplicates


def get_logical_operator(circuit: stim.Circuit):
    # todo
    pass


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
    genc = stim.Circuit.generated('surface_code:rotated_memory_z',
                                  distance=3,
                                  rounds=4,
                                  after_clifford_depolarization=cparams.two_qubit_depolarization_rate,
                                  before_measure_flip_probability=0.2,
                                  after_reset_flip_probability=0.3
                                  )
    get_matching_matrix(genc)
    # print(reversed(genc))
    print(to_scheduled_circuit(genc, cparams))

#        print(type(inst), '\t', inst)
