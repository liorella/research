import numpy as np
from qiskit import *
from qiskit import Aer
from math import pi

#Changing the simulator
backend = Aer.get_backend('unitary_simulator')

#The circuit without measurement

q = QuantumRegister(5,'q')
c = ClassicalRegister(5,'c')
qc = QuantumCircuit(q,c)
qc.cx(q[2],q[1])
qc.cx(q[4],q[3])
qc.rz(-pi/4,q[1])
qc.rz(-pi/4,q[3])
qc.cx(q[0],q[1])
qc.cx(q[0],q[3])
qc.rz(pi/4,q[1])
qc.rz(pi/4,q[3])
qc.cx(q[1],q[2])
qc.cx(q[3],q[4])

#job execution and getting the result as an object
job = execute(qc, backend, shots=8192)
result = job.result()

#get the unitary matrix from the result object
a=result.get_unitary(qc, decimals=3)
