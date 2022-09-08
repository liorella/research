import stim

c=stim.Circuit("""
H 0 
CNOT 0 1
CNOT 0 2
M 0 1 2 
""")#hadamard on qubit 0 then #CNOT between 0 and 1 then #measure both

print(c.compile_sampler().sample(1)) ##measurement result


def rep_code(distance, rounds, noise):
    circuit=stim.Circuit()
    qubits = range(2*distance+1)
    data = qubits[::2] #data qubits
    measure = qubits[1::2]
    #circuit.append_operation("X_ERROR", data,noise) #one example of noise
    #for m in measure:
    #    circuit.append_operation("CNOT",[m-1,m])
    pairs1=qubits[:-1]
    circuit.append_operation("CNOT", pairs1)
    #for m in measure:
     #   circuit.append_operation("CNOT", [m + 1, m])
    circuit.append_operation("DEPOLARIZE2", pairs1, noise)
    pairs2=qubits[1:][::-1]
    circuit.append_operation("CNOT", pairs2)
    circuit.append_operation("DEPOLARIZE2", pairs2, noise)

    circuit.append_operation("DEPOLARIZE1", qubits, noise)
    circuit.append_operation("MR", measure)
    for k in range(len(measure)):
        circuit.append_operation("DETECTOR", [stim.target_rec(-1), stim.target_rec(-1-k-distance)]) #looking at measurement

    full_circuit = stim.Circuit()
    full_circuit.append_operation("M", measure)
    full_circuit += circuit*rounds
    return full_circuit

 # print(rep_code(3, 2, 0.1))

# the CX shows the controlled not, read in pairs (qubit 6 control 5)

def shot(circuit):
    sample=circuit.compile_detector_sampler().sample(1)[0]*1 #multiplied with 1 to convert to integer
    print("".join("_1"[e] for e in sample))


import shutil

nc=shutil.get_terminal_size().columns #

print(shot(rep_code(nc,20,0.01)))


