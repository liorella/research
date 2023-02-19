import numpy as np

import stim


## building circuit
def cat_state_prep(circ, anc):
    circ.append('H', anc[-1])
    circ.append('TICK')
    for i in (anc[:-1]):
        circ.append('CX', [anc[-1], i])
        circ.append('TICK')



def measure_stabilizer(stabilizer: str,
                       circ,
                       anc
                       ):
    cat_state_prep(circ, anc)
    if stabilizer == "S1":
        circ.append('CX', [anc[0], 1, anc[1], 2, anc[2], 4, anc[3], 3])
    elif stabilizer == "S2":
        circ.append('CX', [anc[0], 2, anc[1], 5, anc[2], 6, anc[3], 3])
    elif stabilizer == "S3":
        circ.append('CX', [anc[0], 4, anc[1], 3, anc[2], 6, anc[3], 7])
    elif stabilizer == "S4":
        circ.append('CZ', [anc[0], 1, anc[1], 2, anc[2], 4, anc[3], 3])
    elif stabilizer == "S5":
        circ.append('CZ', [anc[0], 2, anc[1], 5, anc[2], 6, anc[3], 3])
    elif stabilizer == "S6":
        circ.append('CZ', [anc[0], 4, anc[1], 3, anc[2], 6, anc[3], 7])
    circ.append('TICK')
    circ.append('H', anc)
    circ.append('TICK')
    circ.append('MR', anc)
    circ.append("DETECTOR", [stim.target_rec(- 1), stim.target_rec(- 2), stim.target_rec(-3), stim.target_rec(-4), stim.target_rec(-5)])
    circ.append('TICK')


def measure_data(circ, data_qubits):
    circ.append("M", data_qubits)
    circ.append("OBSERVABLE_INCLUDE",
                [stim.target_rec(- 1), stim.target_rec(- 2), stim.target_rec(-3), stim.target_rec(-4),
                 stim.target_rec(-5), stim.target_rec(-6), stim.target_rec(-7)], 0)

def propagate_state(circ, data_qubits, original, anc):
    circ.append('CX', [original, data_qubits[0]])
    circ.append("TICK")
    circ.append('CX', [1, anc[1]])
    circ.append("TICK")
    circ.append('CX', [anc[1], 2])
    circ.append("TICK")
    circ.append('CX', [anc[1], anc[0], 2, anc[2]])
    circ.append("TICK")
    circ.append('CX', [anc[1], 4, anc[0], anc[4], anc[2], 5])
    circ.append("TICK")
    circ.append('CX', [4, anc[3], anc[4], 7, anc[2], 3])
    circ.append("TICK")
    circ.append('CX', [anc[3], 6])
    circ.append('H', anc)
    circ.append('MR', anc)
    # here a conditioned correction to the data qubits is needed after th ancilla are measured

##
qubits = range(14)
original = qubits[0]
target = qubits[-1]
data_qubits = list(qubits[1:8])
anc = list(qubits[8:13])
circ = stim.Circuit()

circ.append("R", qubits)
circ.append("RX", original)
circ.append("TICK")
# circ.append('X', original)

propagate_state(circ, data_qubits, original, anc)

measure_stabilizer("S1", circ, anc)
measure_stabilizer("S2", circ, anc)
measure_stabilizer("S3", circ, anc)
measure_stabilizer("S4", circ, anc)
measure_stabilizer("S5", circ, anc)
measure_stabilizer("S6", circ, anc)


 circ.append("H", data_qubits)
 # circ.append("H", original)

circ.append("TICK")

measure_data(circ, data_qubits)
circ.append("M", original)
circ.append("DETECTOR",
            [stim.target_rec(- 1), stim.target_rec(- 2), stim.target_rec(-3), stim.target_rec(-4),
             stim.target_rec(-5), stim.target_rec(-6), stim.target_rec(-7), stim.target_rec(-8)])

# print(circ.diagram())

##
sampler = circ.compile_detector_sampler()

syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)

print(sum(syndrome))

print(sum(actual_observables))
## play
circ2 = stim.Circuit()
circ2.append("RZ", [0, 1])
circ2.append("H", 0)
circ2.append("CX", [0, 1])
circ2.append("MX", [0])
circ2.append("OBSERVABLE_INCLUDE", [stim.target_rec(- 1)], 0)

sampler2 = circ2.compile_detector_sampler()

syndrome2, actual_observables2 = sampler2.sample(shots=1000, separate_observables=True)

print(sum(syndrome2))

print(sum(actual_observables2))
