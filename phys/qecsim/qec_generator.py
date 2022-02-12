from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from dataclasses_json import dataclass_json
from matplotlib import pyplot as plt
from quantumsim import circuit as circuit
from scipy.linalg import toeplitz


@dataclass_json
@dataclass
class CircuitParams:
    t1: float
    t2: float
    single_qubit_gate_duration: float
    two_qubit_gate_duration: float
    single_qubit_depolarization_rate: float
    two_qubit_depolarization_rate: float
    meas_duration: float
    reset_duration: float
    reset_latency: float
    meas_induced_dephasing_enhancement: float = 0.0


class QECGenerator(metaclass=ABCMeta):
    def __init__(self, distance: int, circuit_params: CircuitParams):
        self.sampler = circuit.BiasedSampler(readout_error=0, seed=42, alpha=1)
        self.params = circuit_params
        self.distance = distance
        self.qubits = []
        self.cbits = []

    @abstractmethod
    def generate_state_encoder(self, state, plot):
        """
        prepare an encoded state along one of the cardinal points in the logical Bloch sphere

        :param state: either '0', '1', '+', '-', 'i', '-i'
        :param plot: plot circuit
        :return: a circuit object which encodes the state
        """
        pass

    @abstractmethod
    def generate_stabilizer_round(self, final_round, injected_errors, plot):
        pass

    def init_circuit(self, name=None):
        if name is None:
            name = "unnamed circuit"
        c = circuit.Circuit(name)
        for q in self.qubits:
            c.add_qubit(q)
        for cb in self.cbits:
            c.add_qubit(cb)
        return c

    def generate_bitflip_error(self, qubit_names: Iterable[str], plot=False) -> circuit.Circuit:
        c = self.init_circuit("bitflip error")

        t_start = 0
        t_moment = self.params.single_qubit_gate_duration
        for q in qubit_names:
            c.add_gate(circuit.RotateX(q,
                                       t_start + t_moment / 2,
                                       np.pi))

        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        c.order()

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(15)
            plt.show()

        return c

    @abstractmethod
    def generate_active_reset(self, qubit_names, plot):
        pass

    @abstractmethod
    def generate_measure_data(self, plot):
        pass

    @property
    def cbit_names(self):
        return [cb.name for cb in self.cbits]

    @property
    def register_names(self):
        return [q.name for q in self.qubits] + [cb.name for cb in self.cbits]

    @property
    def matching_matrix(self):
        return toeplitz([1] + [0] * (self.distance - 1), [1, 1] + [0] * (self.distance - 1))

    @property
    def qubit_names(self):
        return [q.name for q in self.qubits]
