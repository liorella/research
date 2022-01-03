from collections import Iterable

import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import quantumsim.circuit as circuit
import numpy as np
import matplotlib.pyplot as plt

t1 = 10e5
t2 = 10e5
single_qubit_gate_duration = 20
two_qubit_gate_duration = 40
meas_duration = 400
reset_duration = 250
reset_latency = 200

circuits = {
    "no_err": circuit.Circuit(),
    "data_bitflip": circuit.Circuit()
}

sampler = circuit.BiasedSampler(0, seed=42, alpha=1)

qubits = {
    name: circuit.Qubit(name, t1=t1, t2=t2) for name in {"d0", "d1", "a0"}
}

for c in circuits:
    for q in qubits:
        circuits[c].add_qubit(qubits[q])

# todo: make parametrized
cbits = {
    name: circuit.ClassicalBit(name) for name in {"ma0"}
}
for c in circuits:
    for cb in cbits:
        circuits[c].add_qubit(cbits[cb])

# stabilizer round with no errors
########
t_start = 0
t_moment = single_qubit_gate_duration
circuits["no_err"].add_gate(circuit.RotateY("a0",
                                            t_start + t_moment / 2,
                                            np.pi / 2))

t_start += t_moment
t_moment = two_qubit_gate_duration
circuits["no_err"].add_gate(circuit.CPhase("d0", "a0",
                                           t_start + t_moment / 2))

t_start += t_moment
t_moment = two_qubit_gate_duration
circuits["no_err"].add_gate(circuit.CPhase("d1", "a0",
                                           t_start + t_moment / 2))
t_start += t_moment
t_moment = single_qubit_gate_duration
circuits["no_err"].add_gate(circuit.RotateY("a0",
                                            t_start + t_moment / 2,
                                            -np.pi / 2))

t_start += t_moment
t_moment = meas_duration
circuits["no_err"].add_gate(circuit.Measurement("a0",
                                                time=t_start + t_moment / 2,
                                                sampler=sampler,
                                                output_bit="ma0",
                                                ))

t_start += t_moment
circuits["no_err"].add_waiting_gates(0, t_start)
circuits["no_err"].order()

# fig, ax = circuits["no_err"].plot()
# fig.set_figwidth(15)
# plt.show()

# inject data qubit error
##################

t_start = 0
t_moment = single_qubit_gate_duration
circuits["data_bitflip"].add_gate(circuit.RotateX("d0",
                                                  t_start + t_moment / 2,
                                                  np.pi))

t_start += t_moment
circuits["data_bitflip"].add_waiting_gates(0, t_start)
circuits["data_bitflip"].order()


# fig2, ax2 = circuits["data_bitflip"].plot()
# fig2.set_figwidth(15)
# plt.show()

def active_reset_circ(qubits_to_reset: Iterable) -> circuit.Circuit:
    arc = circuit.Circuit()
    for q in qubits:
        arc.add_qubit(qubits[q])
    for cb in cbits:
        arc.add_qubit(cbits[cb])

    t_start = 0
    t_moment = reset_latency

    t_start += t_start
    t_moment = single_qubit_gate_duration

    for q in qubits_to_reset:
        arc.add_gate(circuit.RotateX(q,
                                     t_start + t_moment / 2,
                                     np.pi))
    t_start += t_moment
    arc.add_waiting_gates(0, t_start)
    arc.order()
    # fig3, ax3 = arc.plot()
    # fig.set_figwidth(10)
    # plt.show()
    return arc


# run simulation
########
state = quantumsim.sparsedm.SparseDM(circuits["no_err"].get_qubit_names())
results = []
for i in range(3):
    circuits["no_err"].apply_to(state)
    results.append(state.classical["ma0"])
    if state.classical["ma0"] == 1:
        active_reset_circ(["a0"]).apply_to(state)

circuits["data_bitflip"].apply_to(state)

for i in range(6):
    circuits["no_err"].apply_to(state)
    results.append(state.classical["ma0"])
    if state.classical["ma0"] == 1:
        active_reset_circ(["a0"]).apply_to(state)

circuits["data_bitflip"].apply_to(state)

for i in range(6):
    circuits["no_err"].apply_to(state)
    results.append(state.classical["ma0"])
    if state.classical["ma0"] == 1:
        active_reset_circ(["a0"]).apply_to(state)

print(results)
