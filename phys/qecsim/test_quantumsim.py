import numpy as np
import quantumsim.sparsedm
from quantumsim import circuit
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def test_readout_error():
    sampler = circuit.BiasedSampler(0, 1, seed=42)
    num_reps = int(1e4)
    # measure_duration = 400
    single_qubit_gate_duration = 20
    meas_1_error_vec = []
    meas_0_error_vec = []
    meas_duration_vec = np.linspace(0, 1000, 20)
    t1 = 15e3
    t2 = 19e3
    for meas_duration in tqdm(meas_duration_vec):

        meas_1_error = 0
        meas_0_error = 0
        for k in range(2 * num_reps):
            state = quantumsim.sparsedm.SparseDM(['q1', 'm1'])
            c = circuit.Circuit()
            q1 = circuit.Qubit('q1', t1=t1, t2=t2)
            c1 = circuit.ClassicalBit('m1')
            for q in [q1, c1]:
                c.add_qubit(q)

            t_start = 0
            t_moment = single_qubit_gate_duration
            if k < num_reps:
                c.add_gate(circuit.RotateX('q1', t_start + t_moment / 2, np.pi))
            else:
                t_moment = single_qubit_gate_duration
                c.add_gate(circuit.RotateX('q1', t_start + t_moment / 2, np.pi * 2))  # otherwise it'll be classical

            t_start += t_moment
            t_moment = meas_duration
            c.add_gate(circuit.Measurement('q1', t_start + t_moment / 2, sampler, output_bit='m1'))

            t_start += t_moment
            c.add_waiting_gates(0, t_start)
            c.order()

            # run simulation
            c.apply_to(state)
            if k < num_reps:
                if state.classical['m1'] == 0:
                    meas_1_error += 1
            else:
                if state.classical['m1'] == 1:
                    meas_0_error += 1

        meas_0_error /= num_reps
        meas_1_error /= num_reps
        meas_0_error_vec.append(meas_0_error)
        meas_1_error_vec.append(meas_1_error)
    meas_duration_vec = np.array(meas_duration_vec)
    plt.plot(meas_duration_vec, meas_1_error_vec, label="measure 1 error")
    plt.plot(meas_duration_vec, meas_0_error_vec, label="measure 0 error")
    plt.title(f'prepare |1>, measure |0> prob, T1 = {t1}, T2 = {t2}')
    plt.xlabel('measurement duration')
    plt.ylabel('measurement error')
    plt.legend()
    plt.show()
