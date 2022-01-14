import numpy as np
import quantumsim.sparsedm
from quantumsim import circuit
import matplotlib.pyplot as plt


def test_is_from_joint_distribution():
    """
    this tests that quantumsim samples from the joint distribution over all qubits by creating a bell state
    and measuring its statistics
    """
    sampler = circuit.BiasedSampler(0, 1, seed=42)
    hist_x = []
    hist_y = []
    for _ in range(100):
        state = quantumsim.sparsedm.SparseDM(['q1', 'q2', 'm1', 'm2'])
        c = circuit.Circuit()
        q1, q2 = circuit.Qubit('q1'), circuit.Qubit('q2')
        c1, c2 = circuit.ClassicalBit('m1'), circuit.ClassicalBit('m2')
        for q in [q1, q2, c1, c2]:
            c.add_qubit(q)
        c.add_gate(circuit.RotateY('q1', 10, np.pi / 2))
        c.add_gate(circuit.CNOT('q2', 'q1', 60))
        c.add_gate(circuit.Measurement('q1', 120, sampler, output_bit='m1'))
        c.add_gate(circuit.Measurement('q2', 120, sampler, output_bit='m2'))
        c.add_waiting_gates(0, 200)
        c.order()
        c.apply_to(state)
        m1, m2 = state.classical['m1'], state.classical['m2']
        print(m1, m2)
        assert m1 == m2, "measurement is not from the joint distribution"
        hist_x.append(m1)
        hist_y.append(m2)
    # plt.hist2d(hist_x, hist_y, bins=2)
    # plt.show()
