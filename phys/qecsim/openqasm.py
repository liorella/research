from qiskit import QuantumCircuit
import numpy as np

## factoring 15  example
qc= QuantumCircuit(6)


qc.h(0)
qc.h(1)
qc.x(2)
qc.ccx(0,2,5)
qc.cswap(0,2,5)
qc.cswap(1,2,4)
qc.cswap(1,3,5)
qc.swap(0,1)
qc.h(0)
qc.cp(np.pi/2,1,0)
qc.h(1)
qc.measure_all

qc.draw()
##
qc.qasm(formatted=True,filename='factoring15.qasm')