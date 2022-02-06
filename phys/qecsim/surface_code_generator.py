import numpy as np
from matplotlib import pyplot as plt
from quantumsim import circuit
import stim
from qecsim.qec_generator import QECGenerator, CircuitParams
from qecsim.rep_code_generator import noisy_c_phase_ptm


def extract_qubits(circ):
    qubits = {}
    for inst in circ:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == 'QUBIT_COORDS':
            qubits[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    return qubits


def extract_ancillas(circ):
    ancillas = set()
    for inst in circ:
        if inst.name == 'MR':  # a qubit will be measured and reset iff it's an ancilla
            ancillas = ancillas.union(t.value for t in inst.targets_copy())
    return sorted(list(ancillas))


def to_str(lst):
    return [str(k) for k in lst]


class SurfaceCodeGenerator(QECGenerator):
    def __init__(self, distance: int, circuit_params: CircuitParams):
        if distance != 3:
            raise NotImplementedError("currently only d=3 is supported")
        super().__init__(distance, circuit_params)
        self._stim_circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=3)
        self.qubit_coords_map = extract_qubits(self._stim_circ)
        self.ancillas = extract_ancillas(self._stim_circ)
        self.cbits = [
            circuit.ClassicalBit("m" + str(i)) for i in self.qubit_coords_map.keys()
        ]
        self.qubits = [
            circuit.Qubit(str(i), t1=self.params.t1, t2=self.params.t2) for i in self.qubit_coords_map.keys()
        ]
        self.logical_x_qubits = [3, 10, 17]  # todo: generate from code for any distance
        self.logical_z_qubits = [1, 3, 5]

    def generate_state_encoder(self, state, plot=False) -> circuit.Circuit:
        """
        prepare an encoded state along one of the cardinal points in the logical Bloch sphere

        Note: This will prepare an eigenstate of the relevant logical operator in its product basis.
        After this step, we need to do a stabilizer round, and only after this round the state will be prepared.

        :param state: either '0', '1'
        :param plot: plot circuit
        :return: a circuit object which encodes the state
        """
        c = self.init_circuit(f"state encoder for |{state}>")

        allowed_states = ['0', '1']
        if state not in allowed_states:
            raise ValueError(f'state must be one of {allowed_states}')
        if state == '0':
            return c
        if state == '1':
            return self.generate_logical_x(plot)

    def generate_stabilizer_round(self, final_round=False, injected_errors=None, plot=False) -> circuit.Circuit:
        """
            H 2 11 16 25
            TICK
            CX 2 3 16 17 11 12 15 14 10 9 19 18
            TICK
            CX 2 1 16 15 11 10 8 14 3 9 12 18
            TICK
            CX 16 10 11 5 25 19 8 9 17 18 12 13
            TICK
            CX 16 8 11 3 25 17 1 9 10 18 5 13
            TICK
            H 2 11 16 25
        """
        c = self.init_circuit("surface code round")
        t_dur1 = self.params.single_qubit_gate_duration
        t_dur2 = self.params.two_qubit_gate_duration

        t_start = 0
        t_moment = 2 * t_dur1
        c = self.add_hadamards(c,
                               to_str([2, 11, 16, 25]),
                               t_start)

        t_start += t_moment
        t_moment = 2 * t_dur1 + t_dur2
        c = self.add_cnots(c,
                           to_str([2, 3, 16, 17, 11, 12, 15, 14, 10, 9, 19, 18]),
                           t_start)

        t_start += t_moment
        t_moment = 2 * t_dur1 + t_dur2
        c = self.add_cnots(c,
                           to_str([2, 1, 16, 15, 11, 10, 8, 14, 3, 9, 12, 18]),
                           t_start)

        t_start += t_moment
        t_moment = 2 * t_dur1 + t_dur2
        c = self.add_cnots(c,
                           to_str([16, 10, 11, 5, 25, 19, 8, 9, 17, 18, 12, 13]),
                           t_start)

        t_start += t_moment
        t_moment = 2 * t_dur1 + t_dur2
        c = self.add_cnots(c,
                           to_str([16, 8, 11, 3, 25, 17, 1, 9, 10, 18, 5, 13]),
                           t_start)

        t_start += t_moment
        t_moment = 2 * t_dur1
        c = self.add_hadamards(c,
                               to_str([2, 11, 16, 25]),
                               t_start)

        t_start += t_moment
        t_moment = self.params.meas_duration
        qubits_to_measure = self.qubit_names if final_round else [str(q) for q in self.ancillas]
        cbits_to_measure = self.cbit_names if final_round else [f'm{q}' for q in self.ancillas]
        for q, cb in zip(qubits_to_measure, cbits_to_measure):
            c.add_gate(circuit.Measurement(q,
                                           time=t_start + t_moment / 2,
                                           sampler=self.sampler,
                                           output_bit=cb
                                           ))

        c.order()
        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(15)
            plt.show()
        return c

    def add_hadamards(self, c, qubits, t_start):
        """H  X RY-pi/2  RYpi/2 X """
        t_dur1 = self.params.single_qubit_gate_duration
        for q in qubits:
            c.add_gate(circuit.RotateX(q,
                                       t_start + t_dur1 / 2,
                                       -np.pi,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate,
                                       dephasing_axis=self.params.single_qubit_depolarization_rate))

            c.add_gate(circuit.RotateY(q,
                                       t_start + 3 * t_dur1 / 2,
                                       -np.pi / 2,  # this sign checks out when I checked the DM of a single qubit
                                       dephasing_angle=self.params.single_qubit_depolarization_rate,
                                       dephasing_axis=self.params.single_qubit_depolarization_rate))
        return c

    def add_cnots(self, c, qubit_pairs,
                  t_start):

        """ CNOT  I RYpi/2 CPh I RY-pi/2 """
        t_dur1 = self.params.single_qubit_gate_duration
        t_dur2 = self.params.two_qubit_gate_duration

        for q1, q2 in zip(qubit_pairs[::2], qubit_pairs[1::2]):
            c.add_gate(circuit.RotateY(q2,
                                       t_start + t_dur1 / 2,
                                       np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate,
                                       dephasing_axis=self.params.single_qubit_depolarization_rate))

            c.add_gate(circuit.TwoPTMGate(q1, q2,
                                          noisy_c_phase_ptm(self.params.two_qubit_depolarization_rate),
                                          t_start + t_dur1 + t_dur2 / 2,
                                          ))

            c.add_gate(circuit.RotateY(q2,
                                       t_start + t_dur2 + 3 * t_dur1 / 2,
                                       -np.pi / 2,
                                       dephasing_angle=self.params.single_qubit_depolarization_rate,
                                       dephasing_axis=self.params.single_qubit_depolarization_rate))
        return c

    def generate_active_reset(self, qubit_names, plot=False) -> circuit.Circuit:
        pass

    def generate_measure_data(self, plot=False) -> circuit.Circuit:
        pass

    def generate_logical_x(self, plot=False):
        c = self.init_circuit("logical X")

        t_start = 0
        t_moment = self.params.single_qubit_gate_duration
        for q in self.logical_x_qubits:
            c.add_gate(circuit.RotateX(q,
                                       t_start + t_moment / 2,
                                       np.pi))

        c.order()
        t_start += t_moment
        c.add_waiting_gates(0, t_start)
        if plot:
            fig, ax = c.plot()
            fig.set_figwidth(15)
            plt.show()
        return c


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
    generator = SurfaceCodeGenerator(3, cparams)
    generator.generate_stabilizer_round(plot=True)

    plt.show()
