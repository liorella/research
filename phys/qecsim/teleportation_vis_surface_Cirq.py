import cirq


############ sensitive to bit flip

circuit = cirq.Circuit()

#defining qubits
source = cirq.GridQubit(0, 0)
target = cirq.GridQubit(2, 2)
d1 = cirq.GridQubit(1, 0)
d2 = cirq.GridQubit(0, 1)
d3 = cirq.GridQubit(2, 1)
d4 = cirq.GridQubit(1, 2)
anc = cirq.GridQubit(1, 1)

circuit.append([cirq.H(source), cirq.H(d1), cirq.H(d2), cirq.H(d3), cirq.H(d4), cirq.H(anc), cirq.H(target)])
circuit.append([cirq.CZ(target, d3), cirq.CZ(anc, d1)])
circuit.append([cirq.CZ(target, d4), cirq.CZ(anc, d2)])
circuit.append([cirq.CZ(source, d1), cirq.CZ(anc, d3)])
circuit.append([cirq.CZ(anc, d4)])
circuit.append([cirq.H(anc), cirq.H(target)])
circuit.append([cirq.measure(anc)])

# if the measurement is 0, don't do anything. if the measurement id 1, correct qubit d2 with the command
circuit.append([cirq.X(d2)])

# from here on continue until the measurement
circuit.append([cirq.CZ(source, d2)])
circuit.append([cirq.H(source), cirq.H(d1), cirq.H(d2), cirq.H(d3), cirq.H(d4)])
circuit.append([cirq.measure(source), cirq.measure(d1), cirq.measure(d2), cirq.measure(d3), cirq.measure(d4)])

#conditional corrections
# if XOR(d1,d3) != XOR(d2,d4) : error detected, don't use results
# elsif XOR(d1,d3) circuit.append([cirq.Z(target)])
# if source circuit.append([cirq.X(target)])

print(circuit)


############ sensitive to phase flip

circuit = cirq.Circuit()

#defining qubits
source = cirq.GridQubit(0, 0)
target = cirq.GridQubit(2, 2)
d1 = cirq.GridQubit(1, 0)
d2 = cirq.GridQubit(0, 1)
d3 = cirq.GridQubit(2, 1)
d4 = cirq.GridQubit(1, 2)
anc = cirq.GridQubit(1, 1)

circuit.append([cirq.H(d1), cirq.H(d2), cirq.H(d3), cirq.H(d4), cirq.H(anc), cirq.H(target)])
circuit.append([cirq.CZ(target, d3), cirq.CZ(anc, d1)])
circuit.append([cirq.CZ(target, d4), cirq.CZ(anc, d2)])
circuit.append([cirq.CZ(source, d1), cirq.CZ(anc, d3)])
circuit.append([cirq.CZ(anc, d4)])
circuit.append([cirq.H(anc), cirq.H(target)])
circuit.append([cirq.measure(anc)])

# if the measurement is 0, don't do anything. if the measurement id 1, correct qubit d2 with the command
circuit.append([cirq.X(d2)])

# from here on continue until the measurement
circuit.append([cirq.CZ(source, d2)])
circuit.append([cirq.H(source), cirq.H(d1), cirq.H(d2), cirq.H(d3), cirq.H(d4)])
circuit.append([cirq.measure(source), cirq.measure(d1), cirq.measure(d2), cirq.measure(d3), cirq.measure(d4)])

#conditional corrections
# if XOR(d1,d3) != XOR(d2,d4) : error detected, don't use results
# elsif XOR(d1,d3) circuit.append([cirq.X(target)])
# if source circuit.append([cirq.Z(target)])

print(circuit)
