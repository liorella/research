from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from quantumsim import circuit as circuit
from scipy.linalg import toeplitz


@dataclass
class CircuitParams:
    t1: float
    t2: float
    single_qubit_gate_duration: float
    two_qubit_gate_duration: float
    meas_duration: float
    reset_duration: float
    reset_latency: float


class RepCodeGenerator:
    def __init__(self, distance: int, circuit_params: CircuitParams):
        self.distance = distance
        self.params = circuit_params
        self.sampler = circuit.BiasedSampler(readout_error=0, seed=42, alpha=1)

        self.qubits = [
            circuit.Qubit(str(i), t1=self.params.t1, t2=self.params.t2) for i in range(2 * self.distance + 1)
        ]
        self.cbits = [
            circuit.ClassicalBit("m" + str(i)) for i in range(2 * self.distance + 1)
        ]

    @property
    def register_names(self):
        return [q.name for q in self.qubits] + [cb.name for cb in self.cbits]

    @property
    def qubit_names(self):
        return [q.name for q in self.qubits]

    @property
    def cbit_names(self):
        return [cb.name for cb in self.cbits]

    @property
    def matching_matrix(self):
        return toeplitz([1] + [0] * (self.distance - 1), [1, 1] + [0] * (self.distance - 1))

    def generate_stabilizer_round(self,
                                  final_round=False,
                                  injected_errors=None,
                                  plot=False) -> circuit.Circuit:
        if injected_errors is not None:
            raise NotImplementedError
        # todo: inject errors inside the stabilizer round
        # stabilizer round with no errors
        ########
        c = self.init_circuit("stabilizer round")

        t_start = 0
        t_moment = self.params.single_qubit_gate_duration

        for q in self.qubit_names[1::2]:
            c.add_gate(circuit.RotateY(q,
                                       t_start + t_moment / 2,
                                       np.pi / 2))

        t_start += t_moment
        t_moment = self.params.two_qubit_gate_duration
        for q1, q2 in zip(self.qubit_names[:-1:2], self.qubit_names[1::2]):
            c.add_gate(circuit.CPhase(q1, q2,
                                      t_start + t_moment / 2))

        t_start += t_moment
        t_moment = self.params.two_qubit_gate_duration
        for q1, q2 in zip(self.qubit_names[2::2], self.qubit_names[1::2]):
            c.add_gate(circuit.CPhase(q1, q2,
                                      t_start + t_moment / 2))

        t_start += t_moment
        t_moment = self.params.single_qubit_gate_duration
        for q in self.qubit_names[1::2]:
            c.add_gate(circuit.RotateY(q,
                                       t_start + t_moment / 2,
                                       -np.pi / 2))

        t_start += t_moment
        t_moment = self.params.meas_duration
        qubits_to_measure = self.qubit_names if final_round else self.qubit_names[1::2]
        cbits_to_measure = self.cbit_names if final_round else self.cbit_names[1::2]
        for q, cb in zip(qubits_to_measure, cbits_to_measure):
            c.add_gate(circuit.Measurement(q,
                                           time=t_start + t_moment / 2,
                                           sampler=self.sampler,
                                           output_bit=cb
                                           ))

        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        c.order()

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(15)
            plt.show()
        return c

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

    def generate_active_reset(self, qubit_names: Iterable[str], plot=False) -> circuit.Circuit:
        c = self.init_circuit()
        t_start = 0
        t_moment = self.params.reset_latency
        # only wait for the latency period

        t_start += t_start
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
            fig.set_figwidth(10)
            plt.show()
        return c

    def generate_measure_data(self, plot=False):
        c = self.init_circuit()
        t_start = 0
        t_moment = self.params.meas_duration

        for q, cb in zip(self.qubit_names[::2], self.cbit_names[::2]):
            c.add_gate(circuit.Measurement(q,
                                           time=t_start + t_moment / 2,
                                           sampler=self.sampler,
                                           output_bit=cb
                                           ))
        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        c.order()

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(10)
            plt.show()
        return c

    def generate_logical_X(self, plot=False):
        c = self.init_circuit()
        t_start = 0
        t_moment = self.params.single_qubit_gate_duration

        for q, cb in zip(self.qubit_names[::2], self.cbit_names[::2]):
            c.add_gate(circuit.RotateX(q,
                                       time=t_start + t_moment / 2,
                                       angle=np.pi
                                       ))
        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        c.order()

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(10)
            plt.show()
        return c
