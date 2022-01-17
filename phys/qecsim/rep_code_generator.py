from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from quantumsim import circuit as circuit

from qecsim.qec_generator import QECGenerator, CircuitParams


class RepCodeGenerator(QECGenerator):
    def __init__(self, distance: int, circuit_params: CircuitParams):
        super().__init__(distance, circuit_params)
        self.cbits = [
            circuit.ClassicalBit("m" + str(i)) for i in range(2 * self.distance + 1)
        ]
        self.qubits = [
            circuit.Qubit(str(i), t1=self.params.t1, t2=self.params.t2) for i in range(2 * self.distance + 1)
        ]

    def generate_state_encoder(self, state: str, plot=False) -> circuit.Circuit:
        """
        prepare an encoded state along one of the cardinal points in the logical Bloch sphere

        :param state: either '0', '1', '+', '-', 'i', '-i'
        :param plot: plot circuit
        :return: a circuit object which encodes the state
        """
        c = self.init_circuit(f"state encoder for |{state}>")

        t_start = 0
        t_moment = self.params.single_qubit_gate_duration
        allowed_states = ['0', '1', '+', '-', '+i', '-i']
        if state not in allowed_states:
            raise ValueError(f'state must be one of {allowed_states}')
        if state == '0':
            return c
        if state == '1':
            return self.generate_logical_x(plot)
        if state == '+':
            c.add_gate(circuit.RotateY('0',
                                       t_start + t_moment / 2,
                                       np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))
        if state == '-':
            c.add_gate(circuit.RotateY('0',
                                       t_start + t_moment / 2,
                                       -np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        if state == '+i':
            c.add_gate(circuit.RotateX('0',
                                       t_start + t_moment / 2,
                                       -np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        if state == '-i':
            c.add_gate(circuit.RotateX('0',
                                       t_start + t_moment / 2,
                                       np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        for q in self.qubit_names[2::2]:
            c.add_gate(circuit.RotateY(q,
                                       t_start + t_moment / 2,
                                       np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        for q in self.qubit_names[2::2]:
            t_start += t_moment
            t_moment = self.params.two_qubit_gate_duration
            c.add_gate(circuit.NoisyCPhase('0', q,
                                           t_start + t_moment / 2,
                                           dephase_var=self.params.two_qubit_depolarization_rate))

        t_start += t_moment
        t_moment = self.params.single_qubit_gate_duration
        for q in self.qubit_names[2::2]:
            c.add_gate(circuit.RotateY(q,
                                       t_start + t_moment / 2,
                                       -np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        c.order()
        t_start += t_moment
        c.add_waiting_gates(0, t_start)

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(15)
            plt.show()
        return c

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
                                       np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

        t_start += t_moment
        t_moment = self.params.two_qubit_gate_duration
        for q1, q2 in zip(self.qubit_names[2::2], self.qubit_names[1::2]):
            c.add_gate(circuit.NoisyCPhase(q1, q2,
                                           t_start + t_moment / 2,
                                           dephase_var=self.params.two_qubit_depolarization_rate))

        t_start += t_moment
        t_moment = self.params.two_qubit_gate_duration
        for q1, q2 in zip(self.qubit_names[:-1:2], self.qubit_names[1::2]):
            c.add_gate(circuit.NoisyCPhase(q1, q2,
                                           t_start + t_moment / 2,
                                           dephase_var=self.params.two_qubit_depolarization_rate))

        t_start += t_moment
        t_moment = self.params.single_qubit_gate_duration
        for q in self.qubit_names[1::2]:
            c.add_gate(circuit.RotateY(q,
                                       t_start + t_moment / 2,
                                       -np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))

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

    def generate_logical_x(self, plot=False):
        c = self.init_circuit()
        t_start = 0
        t_moment = self.params.single_qubit_gate_duration

        for q, cb in zip(self.qubit_names[::2], self.cbit_names[::2]):
            c.add_gate(circuit.RotateX(q,
                                       time=t_start + t_moment / 2,
                                       angle=np.pi,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate
                                       ))
        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        c.order()

        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(10)
            plt.show()
        return c

    def generate_active_reset(self, qubit_names: Iterable[str], plot=False) -> circuit.Circuit:
        c = self.init_circuit('active reset')
        t_start = 0
        t_moment = self.params.reset_latency
        # only wait for the latency period

        t_start += t_moment
        t_moment = self.params.reset_duration

        for q in qubit_names:
            c.add_gate(circuit.RotateX(q,
                                       t_start + t_moment / 2,
                                       np.pi,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate))
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
