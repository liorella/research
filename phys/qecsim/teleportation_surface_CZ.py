
import numpy as np

import stim
from stim_lib.run_feedback import do_and_get_measure_results
import matplotlib.pyplot as plt
from tqdm import tqdm


## building circuit

def initial_surface(circ, length, data_qubits, anc, p):
    circ.append('QUBIT_COORDS', [0], (0, 1)) # this is the original qubit

    for i in range(2*length):
        circ.append('QUBIT_COORDS', [i+1], (1+2*(i // 2), 2*(i % 2)))
        data_qubits.append(i+1)
    for i in range(length-1): # if length>2 there are more ancillas, we would need to modify accordingly
        circ.append('QUBIT_COORDS', [2*length+1+i], ( 2*(i+1), 1))
        anc.append(2*length+1+i)
    target_qubit=len(data_qubits)+len(anc)+1
    circ.append('QUBIT_COORDS', [target_qubit], (length*2, 1)) # this is the target qubit
    circ.append('R', data_qubits+anc+[target_qubit])
    circ.append('R', [0])
    circ.append("DEPOLARIZE1", data_qubits+anc+[target_qubit]+[0], p/10)
    circ.append('TICK')
    circ.append('H', data_qubits+anc+[target_qubit]+[0])
    circ.append("DEPOLARIZE1", data_qubits+anc+[target_qubit]+[0], p/10)
    circ.append('TICK')


def entangle_target(circ, data_qubits, target_qubit, p):
    circ.append('TICK')
    circ.append('CZ', [target_qubit]+[data_qubits[-1]])
    circ.append("DEPOLARIZE2", [target_qubit]+[data_qubits[-1]], p)
    circ.append('TICK')
    circ.append('CZ', [target_qubit]+[data_qubits[-2]])
    circ.append("DEPOLARIZE2", [target_qubit]+[data_qubits[-2]], p)
    circ.append('TICK')
    circ.append('H', [target_qubit])
    circ.append("DEPOLARIZE1", [target_qubit], p/10)
    circ.append('TICK')

def append_Z_stabilizer(circ, data_qubits, anc, p): #need to modify for larget lengths
    for k in data_qubits:
        circ.append('CZ', anc+[k % 4+1])
        circ.append("DEPOLARIZE2", anc+[k % 4+1], p)
        circ.append('TICK')
    circ.append('H', anc)
    circ.append("DEPOLARIZE1", anc, p/10)
    circ.append('TICK')
    circ.append("DEPOLARIZE1", anc, p)
    circ.append('MR', anc)


def entangle_original_state_circuit(p):
    circ = stim.Circuit()
    circ.append('CZ', [0,1])
    circ.append("DEPOLARIZE2", [0,1], p)
    circ.append('TICK')
    circ.append('CZ', [0,2])
    circ.append("DEPOLARIZE2", [0,2], p)
    circ.append('TICK')
    circ.append('H', 0)
    circ.append("DEPOLARIZE1", 0, p/10)
    circ.append('TICK')
    circ.append("DEPOLARIZE1", 0, p)
    circ.append('M', 0)
    return circ

def data_qubit_mes_circuit(data_qubits, p):
    circ = stim.Circuit()
    circ.append('H', 0)
    circ.append("DEPOLARIZE1", data_qubits, p/10)
    circ.append('TICK')
    circ.append("DEPOLARIZE1", data_qubits, p)
    circ.append('M', data_qubits)
    return circ


def measure_target(target_qubit, p, basis="Z"):
    circ = stim.Circuit()

    circ.append("DEPOLARIZE1", target_qubit, p)
    if basis=="Z":
        circ.append('M', target_qubit)
    elif basis == "X":
        circ.append('MX', target_qubit)
    elif basis == "Y":
        circ.append('MY', target_qubit)
    return circ


def target_correction(target_qubit, p, correction="X"):
    circ = stim.Circuit()
    if correction=="X":
        circ.append('X', target_qubit)
    elif correction=="Z":
        circ.append('Z', target_qubit)
    circ.append("DEPOLARIZE1", target_qubit, p/10)
    circ.append('TICK')
    return circ

def X_correction_circ(data_qubits, p):
    circ = stim.Circuit()
    circ.append('X', data_qubits[1])
    circ.append("DEPOLARIZE1", length, p/10)
    circ.append('TICK')
    return circ


##
length=2
shots = 50000
p_vec = np.logspace(-3, -0.5, num=10)
round_vec= [0, 1, 2]
fail_rate = np.zeros((len(p_vec), len(round_vec)))

discarded = np.zeros((len(p_vec), len(round_vec)))

if length==2:
    target_qubit= 3 * length

for r, rounds in enumerate(round_vec):

    for p_ind, p in tqdm(enumerate(p_vec)):
        circ = stim.Circuit()
        data_qubits = []
        anc = []
        initial_surface(circ, length, data_qubits, anc, p)
        entangle_target(circ, data_qubits, target_qubit, p)
        append_Z_stabilizer(circ, data_qubits, anc, p)

        Z_stabilizer_circuit = stim.Circuit()
        append_Z_stabilizer(Z_stabilizer_circuit, data_qubits, anc, p)

        entangle_original_state_circ=entangle_original_state_circuit(p)
        data_qubit_mes_circ=data_qubit_mes_circuit(data_qubits, p)

        #
        success = 0
        sumilation_count=0
        for shot in range(shots):
            sim = stim.TableauSimulator()
            initialize_anc_mes = do_and_get_measure_results(sim, segment=circ, qubits_to_return=anc)
            syabilizer_mes=[]

            if any(initialize_anc_mes):
                sim.do(X_correction_circ(data_qubits, p))
            original_qubit_mes = do_and_get_measure_results(sim, segment=entangle_original_state_circ, qubits_to_return=[0])
            for j in range(rounds):
                syabilizer_mes.append(do_and_get_measure_results(sim, segment=Z_stabilizer_circuit, qubits_to_return=anc))

            data_qubit_mes = do_and_get_measure_results(sim, segment=data_qubit_mes_circ, qubits_to_return=data_qubits)
            x_log1 = (data_qubit_mes[0]+data_qubit_mes[2])%2
            x_log2 = (data_qubit_mes[1]+data_qubit_mes[3])%2

            if (x_log1 == x_log2) and sum(syabilizer_mes)==0:
                if x_log2:
                    sim.do(target_correction(target_qubit, p, correction="Z"))
                if original_qubit_mes:
                    sim.do(target_correction(target_qubit, p, correction="X"))
                # checking teleportation
                target_state = do_and_get_measure_results(sim, segment=measure_target(target_qubit, p, basis="Z"), qubits_to_return=target_qubit)
                if target_state == 0:
                    success += 1
                sumilation_count+=1


        discarded[p_ind,r]=(1-sumilation_count/shots)
        fail_rate[p_ind,r]=(1-success / sumilation_count)

  np.savez('Tele_Z_Rounds_length2.npz', discarded=discarded, fail_rate=fail_rate, p_vec=p_vec, round_vec=round_vec)
##

fig, ax = plt.subplots(2,1)
for r,rounds in enumerate(round_vec):
    ax[0].plot(p_vec, fail_rate[:,r], 'o', label=f'rounds={rounds}')
    ax[1].plot(p_vec, 100*discarded[:,r], 'o', label=f'rounds={rounds}')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set(xlabel='physical error', ylabel='transmission error')
ax[0].grid(linestyle='--', linewidth=0.2)
ax[0].legend()


ax[1].set_xscale('log')

ax[1].set(xlabel='physical error', ylabel='discarded experiments')
ax[1].grid(linestyle='--', linewidth=0.2)
ax[1].legend()

plt.show()
