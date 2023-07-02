##
import numpy as np

import stim
import pymatching
from matplotlib import pyplot as plt
import networkx as nx

##
circuit_rotated = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.001)

## initial two surfaces in logical X, looking only on X stabilizers
circuit = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
R 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
TICK
H 0 1 2 3 4 5 6 7 8 13 14 15 16 
H 24 25 26 27 28 29 30 31 32 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
DEPOLARIZE2(0.001) 0 9 13 3 6 11 4 10 16 7 14 5
DEPOLARIZE2(0.001) 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
DEPOLARIZE2(0.001) 1 9 13 0 7 11 5 10 16 4 14 2
DEPOLARIZE2(0.001) 25 33 29 34 31 35 37 24 38 26 40 28  
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
DEPOLARIZE2(0.001) 15 3 13 1 4 11 2 10 16 5 8 12
DEPOLARIZE2(0.001) 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
X_ERROR(0.001) 9 10 11 12 13 14 15 16
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
X_ERROR(0.001) 33 34 35 36 37 38 39 40
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]

REPEAT 3 {
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29 
    DEPOLARIZE2(0.001) 0 9 13 3 6 11 4 10 16 7 14 5
    DEPOLARIZE2(0.001) 24 33 37 27 30 35 28 34 40 31 38 29 
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28 
    DEPOLARIZE2(0.001) 1 9 13 0 7 11 5 10 16 4 14 2
    DEPOLARIZE2(0.001) 25 33 29 34 31 35 37 24 38 26 40 28  
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    DEPOLARIZE2(0.001) 15 3 13 1 4 11 2 10 16 5 8 12
    DEPOLARIZE2(0.001) 39 27 28 35 40 29 32 36 26 34 37 25
    TICK
    H 13 14 15 16 37 38 39 40 
    X_ERROR(0.001) 9 10 11 12 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-24]
    DETECTOR rec[-7] rec[-23]
    DETECTOR rec[-6] rec[-22]
    DETECTOR rec[-5] rec[-21]
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
    X_ERROR(0.001) 33 34 35 36 37 38 39 40
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-8] rec[-24]
    DETECTOR rec[-7] rec[-23]
    DETECTOR rec[-6] rec[-22]
    DETECTOR rec[-5] rec[-21]
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]

 
}

H 0 1 2 3 4 5 6 7 8
M 0 1 2 3 4 5 6 7 8
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-18]
DETECTOR rec[-6] rec[-3] rec[-19]
DETECTOR rec[-7] rec[-4] rec[-20]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-21]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]

H 24 25 26 27 28 29 30 31 32
M 24 25 26 27 28 29 30 31 32
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-19]
DETECTOR rec[-6] rec[-3] rec[-20]
DETECTOR rec[-7] rec[-4] rec[-21]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-22]
OBSERVABLE_INCLUDE(1) rec[-7] rec[-8] rec[-9]

''')

model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

##
E=matching.edges() # edges and wieghts
G=matching.to_networkx() #the documentation for networkX graph can be used
options = {
    "font_size": 10,
    "node_size": 200,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
plt.close()
nx.draw_networkx(G, with_labels=True, **options)

## initial two surfaces in logical X, perform surgery,
circuit = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
R 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
TICK
H 0 1 2 3 4 5 6 7 8 13 14 15 16 
H 24 25 26 27 28 29 30 31 32 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
DEPOLARIZE2(0.01) 15 6 13 4 3 11 1 10 16 8 7 12
DEPOLARIZE2(0.01) 39 30 37 28 40 32 25 34 27 35 31 36  
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]

REPEAT 3 {
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29 
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28 
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    DEPOLARIZE2(0.01) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.01) 39 30 37 28 40 32 25 34 27 35 31 36  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    TICK
    H 13 14 15 16 37 38 39 40 
    X_ERROR(0.001) 13 14 15 16 37 38 39 40
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-24]
    DETECTOR rec[-7] rec[-23]
    DETECTOR rec[-6] rec[-22]
    DETECTOR rec[-5] rec[-21]
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
    X_ERROR(0.001) 33 34 35 36 37 38 39 40
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-8] rec[-24]
    DETECTOR rec[-7] rec[-23]
    DETECTOR rec[-6] rec[-22]
    DETECTOR rec[-5] rec[-21]
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
}
TICK
R 17 18 19 20 21 22 23
TICK
H 13 14 15 16 
H 37 38 39 40 
H 20 21 22 23
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29
CX 18 12 21 19 20 17 23 25 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28
CX 23 18 21 8 20 6 19 12  
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
CX 20 18 22 24 17 33 23 26
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
CX 20 7 22 17 18 33 23 19
TICK
H 13 14 15 16 20 21 22 23 37 38 39 40
X_ERROR(0.001) 9 10 11 12 13 14 15 16
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
X_ERROR(0.001) 33 34 35 36 37 38 39 40
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 20 21 22 23
    

REPEAT 3 {    
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    H 20 21 22 23
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29
    CX 18 12 21 19 20 17 23 25 
    DEPOLARIZE2(0.01) 0 9 13 3 6 11 4 10 16 7 14 5
    DEPOLARIZE2(0.01) 24 33 37 27 30 35 28 34 40 31 38 29  
    DEPOLARIZE2(0.01) 18 12 21 19 20 17 23 25   
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28
    CX 23 18 21 8 20 6 19 12  
    DEPOLARIZE2(0.01) 1 9 13 0 7 11 5 10 16 4 14 2
    DEPOLARIZE2(0.01) 25 33 29 34 31 35 37 24 38 26 40 28  
    DEPOLARIZE2(0.01) 23 18 21 8 20 6 19 12  

    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    CX 20 18 22 24 17 33 23 26
    DEPOLARIZE2(0.01) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.01) 39 30 37 28 40 32 25 34 27 35 31 36  
    DEPOLARIZE2(0.01) 20 18 22 24 17 33 23 26  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    CX 20 7 22 17 18 33 23 19
    TICK
    H 13 14 15 16 20 21 22 23 37 38 39 40
    
    X_ERROR(0.001) 9 10 11 12 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-28]
    DETECTOR rec[-7] rec[-27]
    DETECTOR rec[-6] rec[-26]
    DETECTOR rec[-5] rec[-25]
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
    X_ERROR(0.001) 33 34 35 36 37 38 39 40
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-8] rec[-28]
    DETECTOR rec[-7] rec[-27]
    DETECTOR rec[-6] rec[-26]
    DETECTOR rec[-5] rec[-25]
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
    X_ERROR(0.001) 20 21 22 23
    MR 20 21 22 23
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
}
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4]
M 17 18 19





H 0 1 2 3 4 5 6 7 8
M 0 1 2 3 4 5 6 7 8
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-25]
DETECTOR rec[-6] rec[-3] rec[-26]
DETECTOR rec[-7] rec[-4] rec[-27]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-28]
OBSERVABLE_INCLUDE(1) rec[-7] rec[-8] rec[-9]

H 24 25 26 27 28 29 30 31 32
M 24 25 26 27 28 29 30 31 32
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-26]
DETECTOR rec[-6] rec[-3] rec[-27]
DETECTOR rec[-7] rec[-4] rec[-28]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-29]
OBSERVABLE_INCLUDE(2) rec[-7] rec[-8] rec[-9]

''')

model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

# print(circuit.diagram())

##
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)
sum(sum(actual_observables))
num_errors = 0
predicted_observables=np.zeros((syndrome.shape[0],actual_observables.shape[1]))
for i in range(syndrome.shape[0]):
    predicted_observables[i,:] = matching.decode(syndrome[i, :])
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables[i,:])

print(num_errors)  # prints 8
errors=sum(sum(predicted_observables))
print(errors)
##
E = matching.edges()  # edges and wieghts
G = matching.to_networkx()  # the documentation for networkX graph can be used
options = {
    "font_size": 10,
    "node_size": 200,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
plt.close()
nx.draw_networkx(G, with_labels=True, **options)

## initial two surfaces in logical X, perform surgery, and add more stabilizer rounds for two surfaces
circuit = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
R 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
TICK
H 0 1 2 3 4 5 6 7 8 13 14 15 16 
Z 0 1 2 3 4 5 6 7 8
H 24 25 26 27 28 29 30 31 32 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]

REPEAT 3 {
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29 
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28 
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    DEPOLARIZE2(0.01) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.01) 39 30 37 28 40 32 25 34 27 35 31 36  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    TICK
    H 13 14 15 16 37 38 39 40 
    X_ERROR(0.001) 13 14 15 16 37 38 39 40
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
    X_ERROR(0.001) 33 34 35 36 37 38 39 40
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
}
TICK
R 17 18 19 20 21 22 23
TICK
H 13 14 15 16 
H 37 38 39 40 
H 20 21 22 23
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29
CX 18 12 21 19 20 17 23 25 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28
CX 23 18 21 8 20 6 19 12  
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
CX 20 18 22 24 17 33 23 26
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
CX 20 7 22 17 18 33 23 19
TICK
H 13 14 15 16 20 21 22 23 37 38 39 40
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 20 21 22 23


REPEAT 8 {    
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    H 20 21 22 23
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29
    CX 18 12 21 19 20 17 23 25 
    DEPOLARIZE2(0.001) 0 9 13 3 6 11 4 10 16 7 14 5
    DEPOLARIZE2(0.001) 24 33 37 27 30 35 28 34 40 31 38 29  
    DEPOLARIZE2(0.001) 18 12 21 19 20 17 23 25   
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28
    CX 23 18 21 8 20 6 19 12  
    DEPOLARIZE2(0.001) 1 9 13 0 7 11 5 10 16 4 14 2
    DEPOLARIZE2(0.001) 25 33 29 34 31 35 37 24 38 26 40 28  
    DEPOLARIZE2(0.001) 23 18 21 8 20 6 19 12  

    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    CX 20 18 22 24 17 33 23 26
    DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
    DEPOLARIZE2(0.001) 20 18 22 24 17 33 23 26  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    CX 20 7 22 17 18 33 23 19
    TICK
    H 13 14 15 16 20 21 22 23 37 38 39 40
    
    X_ERROR(0.001) 9 10 11 12 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
    X_ERROR(0.001) 33 34 35 36 37 38 39 40
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
    X_ERROR(0.001) 20 21 22 23
    MR 20 21 22 23
    DETECTOR rec[-4] rec[-24]
    DETECTOR rec[-3] rec[-23]
    DETECTOR rec[-2] rec[-22]
    DETECTOR rec[-1] rec[-21]
}
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4]
M 17 18 19

R 9 10 11 12 13 14 15 16
R 33 34 35 36 37 38 39 40
TICK
H 13 14 15 16 
H 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-31]
DETECTOR rec[-7] rec[-30]
DETECTOR rec[-6] rec[-29]
DETECTOR rec[-5] rec[-28] rec[-9] rec[-10]
DETECTOR rec[-4] rec[-27]
DETECTOR rec[-3] rec[-26]
DETECTOR rec[-2] rec[-25]
DETECTOR rec[-1] rec[-24]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-31] rec[-18] rec[-19]
DETECTOR rec[-7] rec[-30]
DETECTOR rec[-6] rec[-29]
DETECTOR rec[-5] rec[-28] 
DETECTOR rec[-4] rec[-27]
DETECTOR rec[-3] rec[-26]
DETECTOR rec[-2] rec[-25]
DETECTOR rec[-1] rec[-24]



REPEAT 8 {
    TICK
    H 13 14 15 16 
    H 37 38 39 40 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29 
    DEPOLARIZE2(0.001) 0 9 13 3 6 11 4 10 16 7 14 5
    DEPOLARIZE2(0.001) 24 33 37 27 30 35 28 34 40 31 38 29   
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28
    DEPOLARIZE2(0.001) 1 9 13 0 7 11 5 10 16 4 14 2
    DEPOLARIZE2(0.001) 25 33 29 34 31 35 37 24 38 26 40 28  
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    DEPOLARIZE2(0.001) 15 6 13 4 3 11 1 10 16 8 7 12
    DEPOLARIZE2(0.001) 39 30 37 28 40 32 25 34 27 35 31 36  
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    DEPOLARIZE2(0.001) 15 3 13 1 4 11 2 10 16 5 8 12
    DEPOLARIZE2(0.001) 39 27 28 35 40 29 32 36 26 34 37 25   
    TICK
    H 13 14 15 16 37 38 39 40 
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
    MR 33 34 35 36 37 38 39 40
    DETECTOR rec[-4] rec[-20]
    DETECTOR rec[-3] rec[-19]
    DETECTOR rec[-2] rec[-18]
    DETECTOR rec[-1] rec[-17]
}

H 0 1 2 3 4 5 6 7 8
M 0 1 2 3 4 5 6 7 8
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-18]
DETECTOR rec[-6] rec[-3] rec[-19]
DETECTOR rec[-7] rec[-4] rec[-20]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-21]
OBSERVABLE_INCLUDE(1) rec[-7] rec[-8] rec[-9]

H 24 25 26 27 28 29 30 31 32
M 24 25 26 27 28 29 30 31 32
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-19]
DETECTOR rec[-6] rec[-3] rec[-20]
DETECTOR rec[-7] rec[-4] rec[-21]
DETECTOR rec[-8] rec[-9] rec[-6] rec[-5] rec[-22]
OBSERVABLE_INCLUDE(2) rec[-7] rec[-8] rec[-9]

''')

model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

# print(circuit.diagram())

##
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)
sum(sum(actual_observables))
num_errors = 0
predicted_observables = np.zeros((syndrome.shape[0], actual_observables.shape[1]))
for i in range(syndrome.shape[0]):
    predicted_observables[i, :] = matching.decode(syndrome[i, :])
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables[i, :])

print(num_errors)  # prints 8
errors = sum(sum(predicted_observables))
print(errors)
##
E = matching.edges()  # edges and wieghts
G = matching.to_networkx()  # the documentation for networkX graph can be used
options = {
    "font_size": 10,
    "node_size": 200,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
plt.close()
nx.draw_networkx(G, with_labels=True, **options)



##
import stim
circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                 distance=3,
                                 rounds=1,
                                 after_clifford_depolarization=0.01,
                                 before_measure_flip_probability=0.3)

circuit.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), 1)


##
circuit = stim.Circuit('''
    R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    H 13 14 15 16 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5
    TICK        
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    TICK
    H 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8]
    DETECTOR rec[-7]
    DETECTOR rec[-6]
    DETECTOR rec[-5]
    H 13 14 15 16 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5
    TICK        
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    TICK
    H 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-16]
    DETECTOR rec[-7] rec[-15]
    DETECTOR rec[-6] rec[-14]
    DETECTOR rec[-5] rec[-13]    
    DETECTOR rec[-4] rec[-12]
    DETECTOR rec[-3] rec[-11]
    DETECTOR rec[-2] rec[-10]
    DETECTOR rec[-1] rec[-9]
    H 0 1 2 3 4 5 6 7 8
    H 9 10 11 12 
    TICK
    CX 9 0 3 13 11 6 10 4 7 16 5 14
    TICK        
    CX 9 1 0 13 11 7 10 5 4 16 2 14
    TICK
    CX 6 15 4 13 11 3 10 1 8 16 12 7
    TICK
    CX 3 15 1 13 11 4 10 2 5 16 12 8
    TICK
    H 9 10 11 12
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-16]
    DETECTOR rec[-7] rec[-15]
    DETECTOR rec[-6] rec[-14]
    DETECTOR rec[-5] rec[-13]    
    DETECTOR rec[-4] rec[-12]
    DETECTOR rec[-3] rec[-11]
    DETECTOR rec[-2] rec[-10]
    DETECTOR rec[-1] rec[-9]
    MX 0 1 2 3 4 5 6 7 8
    DETECTOR rec[-17] rec[-9] rec[-8] 
    DETECTOR rec[-16] rec[-8] rec[-7] rec[-4] rec[-5]
    DETECTOR rec[-15] rec[-6] rec[-3] rec[-5] rec[-2]
    DETECTOR rec[-14] rec[-1] rec[-2] 
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-7]
''')

model = circuit.detector_error_model(decompose_errors=True)
##
circuit = stim.Circuit('''
    DETECTOR rec[-13] rec[-9] rec[-8] rec[-6] rec[-5]
    DETECTOR rec[-12] rec[-4] rec[-7] 
    DETECTOR rec[-11] rec[-6] rec[-3] 
    DETECTOR rec[-10] rec[-1] rec[-2] rec[-5] rec[-4]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
    
    DETECTOR rec[-17] rec[-9] rec[-8] 
    DETECTOR rec[-16] rec[-8] rec[-7] rec[-4] rec[-5]
    DETECTOR rec[-15] rec[-6] rec[-3] rec[-5] rec[-2]
    DETECTOR rec[-14] rec[-1] rec[-2] 
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-7]

''')

## measureing in different basis  - Forcing a +1 wigenstate in the first round when its 50-50
circuit = stim.Circuit('''
    R 0 1 2 3
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2]
    CZ rec[-1] 0
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] 
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]
    MX 0 2
    DETECTOR rec[-1] rec[-2] rec[-3]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]

''')

model = circuit.detector_error_model(decompose_errors=True)

## measureing in different basis
circuit = stim.Circuit('''
    R 0 1 2 3
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2]
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]
    MX 0 2
    DETECTOR rec[-1] rec[-2] rec[-7]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-7]
''')

model = circuit.detector_error_model(decompose_errors=True)

## measureing surgery in a different basis
circuit = stim.Circuit('''
    R 0 1 2 3
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2]
    H 1 3
    CZ 1 0 1 2
    CX 3 0 3 2
    H 1 3
    MR 1 3
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]   
    MX 0 2
    DETECTOR rec[-1] rec[-2] rec[-5]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-5]
    OBSERVABLE_INCLUDE(1) rec[-4]
''')

model = circuit.detector_error_model(decompose_errors=True)



## joint measurement ZZ on 2X2 matrix
circuit = stim.Circuit('''
    R 0 1 2 3 4 5 6 10 11 12 13 14 15 16
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6]
    DETECTOR rec[-4]
    DETECTOR rec[-3]
    DETECTOR rec[-1]
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1] rec[-7]
   
    R 20 
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15 21 22
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-13]
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1]
    
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15 21 22
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-14]
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]
    
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]
    
    M 0 1 2 3 10 11 12 13
    DETECTOR rec[-11] rec[-4] rec[-2]
    DETECTOR rec[-9] rec[-3] rec[-1]
    DETECTOR rec[-14] rec[-8] rec[-6]
    DETECTOR rec[-12] rec[-7] rec[-5]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(1) rec[-5] rec[-6]
 
''')

model = circuit.detector_error_model(decompose_errors=True)

## joint measurement ZZ on 2X2 matrix initiated in XX
circuit = stim.Circuit('''
    R 0 1 2 3 4 5 6 10 11 12 13 14 15 16
    H 0 1 2 3 10 11 12 13
    
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-5]
    DETECTOR rec[-2]
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1] rec[-7]   
  
    R 20 
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-13]
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15 21 22
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-14]
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]
    
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]
    
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1] rec[-7]   

    M 0 1 2 3 10 11 12 13
    DETECTOR rec[-1] rec[-3] rec[-41]
    DETECTOR rec[-2] rec[-4] rec[-43]
    DETECTOR rec[-5] rec[-7] rec[-44]
    DETECTOR rec[-6] rec[-8] rec[-46]
    
    OBSERVABLE_INCLUDE(0) rec[-3] rec[-4] rec[-5] rec[-6] rec[-28]
    
    
    
    
''')

model = circuit.detector_error_model(decompose_errors=True)

## regular teleportation

circuit = stim.Circuit('''
    R 0 1 2
    H 0 1 2
    CZ 2 1 2 0
    H 2 0
    M 2 0
    CX rec[-2] 1
    CZ rec[-1] 1 
    MX 1
    OBSERVABLE_INCLUDE(0) rec[-1]

 ''')
model = circuit.detector_error_model(decompose_errors=True)

## Surface teleportation - success! when initializing in X and measurig in X and performing ZZ surgery
circuit = stim.Circuit('''
    R 0 1 2 3 4 5 6 10 11 12 13 14 15 16
    H 10 11 12 13
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6]
    DETECTOR rec[-4]
    DETECTOR rec[-2]
    CZ rec[-5] 0 
    CX rec[-3] 10 rec[-1] 11
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] 
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] 
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1]   
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1] rec[-7]   

    R 20 
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-13]
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    CX rec[-1] 10 rec[-1] 12
    
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15 21 22
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-14]
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] 
    
    H 5 15 
    CX 2 4 5 3 12 14 15 13 11 20 
    CX 0 4 5 2 10 14 15 12 3 20 
    CX 3 6 5 1 13 16 15 11 10 20
    CX 1 6 5 0 11 16 15 10 2 20 
    H 5 15 21 22
    MR 4 5 6 14 15 16 20 
    DETECTOR rec[-7] rec[-14]
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] 
    
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-13]
    DETECTOR rec[-5] rec[-12]
    DETECTOR rec[-4] rec[-11]
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]
    
       
    H 5 15
    CX 2 4 5 3 12 14 15 13 
    CX 0 4 5 2 10 14 15 12
    CX 3 6 5 1 13 16 15 11
    CX 1 6 5 0 11 16 15 10
    H 5 15
    MR 4 5 6 14 15 16
    DETECTOR rec[-6] rec[-12]
    DETECTOR rec[-5] rec[-11]
    DETECTOR rec[-4] rec[-10]
    DETECTOR rec[-3] rec[-9]
    DETECTOR rec[-2] rec[-8]
    DETECTOR rec[-1] rec[-7]   
    
    MX 0 1 2 3 
    DETECTOR rec[-1] rec[-3] rec[-2] rec[-4] rec[-9]
    CZ rec[-2] 10 rec[-2] 11 rec[-4] 10 rec[-4] 11 
   
    H 15
    CX 12 14 15 13 
    CX 10 14 15 12
    CX 13 16 15 11
    CX 11 16 15 10
    H 15
    MR 14 15 16
    DETECTOR rec[-3] rec[-10]
    DETECTOR rec[-2] rec[-9]
    DETECTOR rec[-1] rec[-8]

    H 15
    CX 12 14 15 13 
    CX 10 14 15 12
    CX 13 16 15 11
    CX 11 16 15 10
    H 15
    MR 14 15 16
    DETECTOR rec[-3] rec[-6]
    DETECTOR rec[-2] rec[-5]
    DETECTOR rec[-1] rec[-4]

    M 10 11 12 13
    DETECTOR rec[-1] rec[-3] rec[-5]
    DETECTOR rec[-2] rec[-4] rec[-7]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]

''')

model = circuit.detector_error_model(decompose_errors=True)

##
circuit = stim.Circuit('''
    R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    H 0 1 2 3 4 5 6 7 8
    H 13 14 15 16 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5
    TICK        
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    TICK
    H 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-4]
    DETECTOR rec[-3]
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    H 13 14 15 16 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5
    TICK        
    CX 1 9 13 0 7 11 5 10 16 4 14 2
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    TICK
    H 13 14 15 16
    MR 9 10 11 12 13 14 15 16
    DETECTOR rec[-8] rec[-16]
    DETECTOR rec[-7] rec[-15]
    DETECTOR rec[-6] rec[-14]
    DETECTOR rec[-5] rec[-13]    
    DETECTOR rec[-4] rec[-12]
    DETECTOR rec[-3] rec[-11]
    DETECTOR rec[-2] rec[-10]
    DETECTOR rec[-1] rec[-9]
    M 0 1 2 3 4 5 6 7 8
    DETECTOR rec[-25] rec[-9] rec[-8] 
    DETECTOR rec[-24] rec[-8] rec[-7] rec[-4] rec[-5]
    DETECTOR rec[-23] rec[-6] rec[-3] rec[-5] rec[-2]
    DETECTOR rec[-22] rec[-1] rec[-2] 
    OBSERVABLE_INCLUDE(0) rec[-25] rec[-9] rec[-8] rec[-23] rec[-6] rec[-3] rec[-5] rec[-2]
''')

model = circuit.detector_error_model(decompose_errors=True)
##
## initial two surfaces in logical X, perform surgery, and add more stabilizer rounds for two surfaces
circuit = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
R 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40

H 13 14 15 16 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-5]
DETECTOR rec[-6]
DETECTOR rec[-7]
DETECTOR rec[-8]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-5]
DETECTOR rec[-6]
DETECTOR rec[-7]
DETECTOR rec[-8]
TICK
H 13 14 15 16 
H 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]


R 17 18 19 
R 20 21 22 23
TICK
H 13 14 15 16 
H 37 38 39 40 
H 20 21 22 23
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29
CX 18 12 21 19 20 17 23 25 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28
CX 23 18 21 8 20 6 19 12  
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
CX 20 18 22 24 17 33 23 26
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
CX 20 7 22 17 18 33 23 19
TICK
H 13 14 15 16 20 21 22 23 37 38 39 40
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-24]
DETECTOR rec[-7] rec[-23]
DETECTOR rec[-6] rec[-22]
DETECTOR rec[-5] rec[-21]
DETECTOR rec[-4] rec[-20]
DETECTOR rec[-3] rec[-19]
DETECTOR rec[-2] rec[-18]
DETECTOR rec[-1] rec[-17]
MR 20 21 22 23


TICK
H 13 14 15 16 
H 37 38 39 40 
H 20 21 22 23
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29
CX 18 12 21 19 20 17 23 25 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28
CX 23 18 21 8 20 6 19 12  

TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
CX 20 18 22 24 17 33 23 26
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
CX 20 7 22 17 18 33 23 19
TICK
H 13 14 15 16 20 21 22 23 37 38 39 40
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-28]
DETECTOR rec[-7] rec[-27]
DETECTOR rec[-6] rec[-26]
DETECTOR rec[-5] rec[-25]
DETECTOR rec[-4] rec[-24]
DETECTOR rec[-3] rec[-23]
DETECTOR rec[-2] rec[-22]
DETECTOR rec[-1] rec[-21]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-28]
DETECTOR rec[-7] rec[-27]
DETECTOR rec[-6] rec[-26]
DETECTOR rec[-5] rec[-25]
DETECTOR rec[-4] rec[-24]
DETECTOR rec[-3] rec[-23]
DETECTOR rec[-2] rec[-22]
DETECTOR rec[-1] rec[-21]
MR 20 21 22 23
DETECTOR rec[-4] rec[-24]
DETECTOR rec[-3] rec[-23]
DETECTOR rec[-2] rec[-22]
DETECTOR rec[-1] rec[-21]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4] rec[-56] rec[-57]  rec[-67] rec[-66] rec[-65] rec[-64] rec[-58] rec[-59]

M 17 18 19


R 9 10 11 12 13 14 15 16
R 33 34 35 36 37 38 39 40
TICK
H 13 14 15 16 
H 37 38 39 40 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
CX 24 33 37 27 30 35 28 34 40 31 38 29 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2
CX 25 33 29 34 31 35 37 24 38 26 40 28 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
CX 39 30 37 28 40 32 25 34 27 35 31 36
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
CX 39 27 28 35 40 29 32 36 26 34 37 25
TICK
H 13 14 15 16 37 38 39 40 
MR 9 10 11 12 13 14 15 16
DETECTOR rec[-8] rec[-31]
DETECTOR rec[-7] rec[-30]
DETECTOR rec[-6] rec[-29]
DETECTOR rec[-5] rec[-28] rec[-9] rec[-10]
DETECTOR rec[-4] rec[-27]
DETECTOR rec[-3] rec[-26]
DETECTOR rec[-2] rec[-25]
DETECTOR rec[-1] rec[-24]
MR 33 34 35 36 37 38 39 40
DETECTOR rec[-8] rec[-31] rec[-18] rec[-19]
DETECTOR rec[-7] rec[-30]
DETECTOR rec[-6] rec[-29]
DETECTOR rec[-5] rec[-28] 
DETECTOR rec[-4] rec[-27]
DETECTOR rec[-3] rec[-26]
DETECTOR rec[-2] rec[-25]
DETECTOR rec[-1] rec[-24]


M 0 1 2 3 4 5 6 7 8
DETECTOR rec[-1] rec[-2] rec[-22]
DETECTOR rec[-2] rec[-3] rec[-5] rec[-6] rec[-23]
DETECTOR rec[-4] rec[-5] rec[-7] rec[-8] rec[-24]
DETECTOR rec[-8] rec[-9] rec[-25]
OBSERVABLE_INCLUDE(1) rec[-3] rec[-6] rec[-9]

M 24 25 26 27 28 29 30 31 32
DETECTOR rec[-1] rec[-2] rec[-23]
DETECTOR rec[-2] rec[-3] rec[-5] rec[-6] rec[-24]
DETECTOR rec[-4] rec[-5] rec[-7] rec[-8] rec[-25]
DETECTOR rec[-8] rec[-9] rec[-26]
OBSERVABLE_INCLUDE(2) rec[-3] rec[-6] rec[-9]

''')

model = circuit.detector_error_model(decompose_errors=True)