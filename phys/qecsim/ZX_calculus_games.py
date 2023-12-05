# %%

import pyzx as zx
qubit_amount = 5
gate_count = 80

#Generate random circuit of Clifford gates
circuit = zx.generate.cliffordT(qubit_amount, gate_count)
#If running in Jupyter, draw the circuit
zx.draw(circuit)
#Use one of the built-in rewriting strategies to simplify the circuit
zx.simplify.full_reduce(circuit)
#See the result
zx.draw(circuit)


Circuit=zx.Circuit
c1 = Circuit(qubit_amount=4)
c1.add_gate("CNOT", 0, 1)

zx.simplify.full_reduce(c1)