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
X_ERROR(0.001) 9 10 11 12 13 14 15 16
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
DETECTOR rec[-4] rec[-27]
DETECTOR rec[-3] rec[-26]
DETECTOR rec[-2] rec[-25]
DETECTOR rec[-1] rec[-24]
MR 33 34 35 36 37 38 39 40
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
model = circuit.detector_error_model(decompose_errors=True)