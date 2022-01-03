import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import quantumsim.circuit as circuit
import numpy as np
import matplotlib.pyplot as plt

t1 = 10e3
t2 = 10e3
single_qubit_gate_duration = 20
two_qubit_gate_duration = 40
meas_duration = 400
reset_duration = 250

circuits = {
    "no_err": circuit.Circuit(),
    "data_bitflip_round": circuit.Circuit()
}

sampler = circuit.BiasedSampler(0, seed=42, alpha=1)

qubits = {
    name: circuit.Qubit(name, t1=t1, t2=t2) for name in {"d0", "d1", "a0"}
}

for c in circuits:
    for q in qubits:
        circuits[c].add_qubit(qubits[q])

# todo: make parametrized
circuits["no_err"].add_qubit(circuit.ClassicalBit("ma0"))

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

fig, ax = circuits["no_err"].plot()
fig.set_figwidth(15)
plt.show()

state = quantumsim.sparsedm.SparseDM(circuits["no_err"].get_qubit_names())
results = []
for i in range(3):
    circuits["no_err"].apply_to(state)
    results.append(state.classical["ma0"])

print(results)
