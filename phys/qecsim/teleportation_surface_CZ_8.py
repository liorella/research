
import numpy as np

import stim
from stim_lib.run_feedback import do_and_get_measure_results
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import inv


## building circuit



def initial_surface(circ, length, data_qubits, anc_z,anc_x, p):
    circ.append('QUBIT_COORDS', [0], (0, 1)) # this is the original qubit
    for i in range(2*length):
        circ.append('QUBIT_COORDS', [i+1], (1+2*(i // 2), 2*(i % 2)))
        data_qubits.append(i+1)

    for i in range(int(length/2)-1): #Z stabilizers
        circ.append('QUBIT_COORDS', [2*length+3*i+1], (2+i*4, 1))
        anc_z.append(2*length+3*i+1)
        circ.append('QUBIT_COORDS', [2 * length + 3*i+2], (4 + i * 4, 3))
        anc_z.append(2 * length + 3*i+2)
        circ.append('QUBIT_COORDS', [2 * length + 3*(i+1)], (4 + i * 4, -1))
        anc_z.append(2 * length + 3*(i+1))
    circ.append('QUBIT_COORDS', [2 * length + 3*(int(length/2)-1)+1], (2+4*(int(length/2)-1), 1))
    anc_z.append(2 * length + 3*(int(length/2)-1)+1)
    if length>2:
        for i in range(int(length/2)-2): #X stabilizers
            circ.append('QUBIT_COORDS', [2 * length + 3*(int(length/2)-1)+2+3*i], (4 + i * 4, 1))
            anc_x.append(2 * length + 3*(int(length/2)-1)+2+3*i)
            circ.append('QUBIT_COORDS', [2 * length + 3*(int(length/2)-1)+1+2+3*i], (6 + i * 4, 3))
            anc_x.append(2 * length + 3*(int(length/2)-1)+1+2+3*i)
            circ.append('QUBIT_COORDS', [2 * length + 3*(int(length/2)-1)+1+3+3*i], (6 + i * 4, -1))
            anc_x.append(2 * length + 3*(int(length/2)-1)+1+3+3*i)
        circ.append('QUBIT_COORDS', [2 * length + 3*(int(length/2)-1)+2+3*(int(length/2)-2)], (4 + (int(length/2)-2) * 4, 1))
        anc_x.append(2 * length + 3*(int(length/2)-1)+2+3*(int(length/2)-2))
    target_qubit=len(data_qubits)+len(anc_z)+len(anc_x)+1
    circ.append('QUBIT_COORDS', [target_qubit], (length*2, 1)) # this is the target qubit
    all_qubits=[0]+data_qubits+anc_z+anc_x+[target_qubit]
    circ.append('R', all_qubits)
    circ.append("DEPOLARIZE1", all_qubits, p/10)
    circ.append('TICK')
    circ.append('H', all_qubits)
    circ.append("DEPOLARIZE1", all_qubits, p/10)
    circ.append('TICK')

##

def append_Z_stabilizer(circ, data_qubits, anc_z, p): # can be optimized to have at most 4 ticks
    for k in range(len(anc_z) // 3):
        circ.append('CZ', [anc_z[k],  data_qubits[4*k]])
        circ.append("DEPOLARIZE2", [anc_z[k],  data_qubits[4*k]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k],  data_qubits[4*k+1]])
        circ.append("DEPOLARIZE2", [anc_z[k],  data_qubits[4*k+1]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k],  data_qubits[4*k+2]])
        circ.append("DEPOLARIZE2", [anc_z[k],  data_qubits[4*k+2]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k],  data_qubits[4*k+3]])
        circ.append("DEPOLARIZE2", [anc_z[k],  data_qubits[4*k+3]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k+1], data_qubits[4 * k+2]])
        circ.append("DEPOLARIZE2", [anc_z[k+1], data_qubits[4 * k+2]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k+1], data_qubits[4 * k + 4]])
        circ.append("DEPOLARIZE2", [anc_z[k+1], data_qubits[4 * k + 4]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k + 2], data_qubits[4 * k + 3]])
        circ.append("DEPOLARIZE2", [anc_z[k + 2], data_qubits[4 * k + 3]], p)
        circ.append('TICK')
        circ.append('CZ', [anc_z[k + 2], data_qubits[4 * k + 5]])
        circ.append("DEPOLARIZE2", [anc_z[k + 2], data_qubits[4 * k + 5]], p)
        circ.append('TICK')
    circ.append('CZ', [anc_z[-1],  data_qubits[-1]])
    circ.append("DEPOLARIZE2", [anc_z[-1],  data_qubits[-1]], p)
    circ.append('TICK')
    circ.append('CZ', [anc_z[-1],  data_qubits[-2]])
    circ.append("DEPOLARIZE2", [anc_z[-1],  data_qubits[-2]], p)
    circ.append('TICK')
    circ.append('CZ', [anc_z[-1],  data_qubits[-3]])
    circ.append("DEPOLARIZE2", [anc_z[-1],  data_qubits[-3]], p)
    circ.append('TICK')
    circ.append('CZ', [anc_z[-1],  data_qubits[-4]])
    circ.append("DEPOLARIZE2", [anc_z[-1],  data_qubits[-4]], p)
    circ.append('TICK')
    circ.append('H', anc_z)
    circ.append("DEPOLARIZE1", anc_z, p / 10)
    circ.append('TICK')
    circ.append("DEPOLARIZE1", anc_z, p)
    circ.append('MR', anc_z)

# def X_correction_mapping(length, results):
#     k=length/2
#     if not(any(results)):
#         return []
#     if k==1:
#         return [1]
#     parity_check_mat=np.zeros((int(1+3*(k-1)),int(2+4*(k-1))))
#     parity_check_mat[0, 0:3] = [1, 1, 1]
#     parity_check_mat[1, 1:4] = [1, 0, 1]
#     parity_check_mat[2, 2:5] = [1, 0, 1]
#     for i in range(int(k-2)):
#         parity_check_mat[(i+1)*3, 3*(i+1):3*(i+2)+1] =[1,1,1,1]
#         parity_check_mat[(i+1)*3+1,3*(i+2)-1:3*(i+2)+2] =[1,0,1]
#         parity_check_mat[(i+1)*3+2,3*(i+2):3*(i+2)+3] =[1,0,1]
#     parity_check_mat[-1, -3::] = [1, 1, 1]
#
#     parity_check_mat2=parity_check_mat
#     to_delete=[]
#     correction_qubits=[1]
#     to_delete.append(3)
#     for i in range(int(k-1)):
#         to_delete.append(4*(i+1))
#         correction_qubits.append((i+1)*4)
#         correction_qubits.append((i+1)*4+1)
#         correction_qubits.append((i + 1) * 4 + 3)
#     parity_check_mat2=np.delete(parity_check_mat2,to_delete,1)
#     correction=parity_check_mat2 @ results %2
#     to_correct=[]
#     for j in range(len(correction)):
#         if correction[j]:
#             to_correct.append(correction_qubits[j])
#     return to_correct



def X_correction_circ(data_qubits, results, p):
    circ = stim.Circuit()
    corrections=X_correction_mapping(data_qubits, results)
    circ.append('X', corrections)
    circ.append("DEPOLARIZE1", anc_z, p / 10)

    return circ

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



##
length=6
shots = 50000
p_vec = np.logspace(-3, -0.5, num=10)
round_vec= [0, 1, 2]
fail_rate = np.zeros((len(p_vec), len(round_vec)))

discarded = np.zeros((len(p_vec), len(round_vec)))


for r, rounds in enumerate(round_vec):

    for p_ind, p in tqdm(enumerate(p_vec)):
        circ = stim.Circuit()
        data_qubits = []
        anc_z = []
        anc_x = []
        initial_surface(circ, length, data_qubits, anc_z, anc_x, p)
        target_qubit = len(data_qubits) + len(anc_z) + len(anc_x) + 1
        append_Z_stabilizer(circ, data_qubits, anc, p)

        entangle_target(circ, data_qubits, target_qubit, p)

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
