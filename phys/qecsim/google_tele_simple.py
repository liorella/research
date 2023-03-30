

import stim
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


num_q_vec=[3,5,7]
shots = 50000
initial_states=["Z","mZ","X","mX","Y","mY"]

## building circuit

def init_circuit(num_q, p,initial_state="Z"):
    circ = stim.Circuit()
    qubits = range(num_q)
    circ.append('R', qubits)
    circ.append("DEPOLARIZE1",qubits, p['init'])
    if initial_state == "X":
        circ.append('H', 0)
        circ.append("DEPOLARIZE1", 0, p['SQ'])
    elif initial_state == "Y":
        circ.append('H_YZ', 0)
        circ.append("DEPOLARIZE1", 0, p['SQ'])
    elif initial_state == "mZ":
        circ.append('X', 0)
        circ.append("DEPOLARIZE1", 0, p['SQ'])
    elif initial_state == "mX":
        circ.append('H', 0)
        circ.append('Z', 0)
        circ.append("DEPOLARIZE1", 0, p['SQ'])
    elif initial_state == "mY":
        circ.append('H_YZ', 0)
        circ.append('Z', 0)
        circ.append("DEPOLARIZE1", 0, p['SQ'])
    return circ

def apply_teleportation(circ,num_q, p,method):
    qubits = range(num_q)
    if method == "swap":
        for i in range(num_q-1):
            circ.append('SWAP', [i,i+1])
            circ.append("DEPOLARIZE2", [i,i + 1], p['swap'])
            circ.append('TICK')

    elif method == "long_range_tele":
        circ.append('TICK')
        circ.append('H', qubits[1:])
        circ.append("DEPOLARIZE1", qubits[1:],p['SQ'])
        circ.append('TICK')
        circ.append('CZ', qubits[1:])
        circ.append("DEPOLARIZE2", qubits[1:], p['CZ'])
        circ.append('TICK')
        circ.append('H', qubits[1:])
        circ.append("DEPOLARIZE1", qubits[1:],p['SQ'])
        circ.append('CZ', qubits[:-1])
        circ.append("DEPOLARIZE2", qubits[:-1], p['CZ'])
        circ.append('TICK')
        circ.append('H', qubits[:-1])
        circ.append("DEPOLARIZE1", qubits[:-1],p['SQ'])
        circ.append('TICK')
        circ.append("X_ERROR", qubits[:-1],p['mes'])
        circ.append('M', qubits[:-1])
        if num_q==3:
            circ.append("DETECTOR", [stim.target_rec(-1)])
            circ.append("DETECTOR", [stim.target_rec(-2)])
        elif num_q==5:
            circ.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
            circ.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4)])
        else:
            circ.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3), stim.target_rec(-5)])
            circ.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4), stim.target_rec(-6)])

def measure_target(circ, num_q, p, initial_state):
    target_qubit=num_q-1
    if initial_state=="mZ":
        circ.append('X', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, p['SQ'])
    elif initial_state == "X":
        circ.append('H', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, p['SQ'])
    elif initial_state == "Y":
        circ.append('H_YZ', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, p['SQ'])
    elif initial_state == "mX":
        circ.append('Z', target_qubit)
        circ.append('H', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, p['SQ'])
    elif initial_state == "mY":
        circ.append('Z', target_qubit)
        circ.append('H_YZ', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, p['SQ'])

    circ.append("X_ERROR", target_qubit, p['mes'])
    circ.append('M', target_qubit)
    circ.append("DETECTOR", [stim.target_rec(-1)])


## single swap value
p = {'SQ': 1.5E-3, 'CZ': 6E-3, 'mes': 2E-2, 'swap': 2E-2, 'init': 1E-3}
n_mat = np.zeros((len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, num_q in enumerate(num_q_vec):
            circ = init_circuit(num_q, p,initial_state=state)
            apply_teleportation(circ,num_q, p, method="swap")
            measure_target(circ, num_q, p, initial_state=state)
            success = 0
            sampler = circ.compile_detector_sampler()
            results = sampler.sample(shots=shots)
            fail_rate = sum(results) / shots
            n_mat[k,m]=2*(1-fail_rate)-1
print(n_mat)

## running over p_swap values

p_swap_vec = np.logspace(-2.5, -1, num=20)
n_mat = np.zeros((len(p_swap_vec), len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, num_q in tqdm(enumerate(num_q_vec)):
        for j, p_swap in enumerate(p_swap_vec):
            p = {'SQ': 1.2E-3, 'CZ': 6E-3, 'mes': 2E-2, 'swap': p_swap, 'init': 1.2E-3}
            circ = init_circuit(num_q, p,initial_state=state)
            apply_teleportation(circ,num_q, p, method="swap")
            measure_target(circ, num_q, p, initial_state=state)
            success = 0
            sampler = circ.compile_detector_sampler()
            results = sampler.sample(shots=shots)
            fail_rate = sum(results) / shots
            n_mat[j,k,m]=2*(1-fail_rate)-1

fig, ax = plt.subplots(2,3,sharex=True, sharey=True)
for m, state in enumerate(initial_states):
    for i, num_q in enumerate(num_q_vec):
        ax[m%2,m//2].plot(p_swap_vec, n_mat[:, i,m], 'o', label=f'{num_q} qubits')
    ax[m%2,m//2].set_title(f'teleportation of {state}')
    ax[m%2,m//2].set_xscale('log')
    if m%2:
        ax[m%2,m//2].set_xlabel('swap error')
    if (m // 2)-1:
        ax[m%2,m//2].set_ylabel('vector length')
    ax[m%2,m//2].grid(linestyle='--', linewidth=0.2)
fig.suptitle('teleportation via swap')

plt.legend()
plt.show()

#np.savez('Tele_via_swap.npz', num_q_vec=num_q_vec, p_swap_vec=p_swap_vec, n_mat=n_mat)
## single teleportation post-processing value
p = {'SQ': 1.5E-3, 'CZ': 6E-3, 'mes': 2E-2, 'swap': 2E-2, 'init': 1E-3}
n_mat = np.zeros((len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, num_q in enumerate(num_q_vec):
            circ = init_circuit(num_q, p,initial_state=state)
            apply_teleportation(circ,num_q, p, method="long_range_tele")
            measure_target(circ, num_q, p, initial_state=state)
            success = 0
            sampler = circ.compile_detector_sampler()
            results = sampler.sample(shots=shots)
            if m==0 or m==1:
                fail=np.logical_xor(results[:,2],results[:,0])
            elif m==2 or m==3:
                fail=np.logical_xor(results[:,2],results[:,1])
            else:
                fail=(results.sum(axis=1))%2
            fail_rate = sum(fail) / shots
            n_mat[k,m]=2*(1-fail_rate)-1

print(n_mat)
## teleportation post-processing

p_CZ_vec = np.logspace(-2.5, -1, num=20)
n_mat = np.zeros((len(p_swap_vec), len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, num_q in tqdm(enumerate(num_q_vec)):
        for j, p_CZ in enumerate(p_CZ_vec):
            p = {'SQ': 1.2E-3, 'CZ': p_CZ, 'mes': 2E-2, 'swap': 2E-2, 'init': 1.2E-3}
            circ = init_circuit(num_q, p,initial_state=state)
            apply_teleportation(circ,num_q, p, method="long_range_tele")
            measure_target(circ, num_q, p, initial_state=state)
            success = 0
            sampler = circ.compile_detector_sampler()
            results = sampler.sample(shots=shots)
            if m==0 or m==1:
                fail=np.logical_xor(results[:,2],results[:,0])
            elif m == 2 or m == 3:
                fail = np.logical_xor(results[:, 2], results[:, 1])
            else:
                fail = (results.sum(axis=1)) % 2
            fail_rate = sum(fail) / shots
            n_mat[j,k,m]=2*(1-fail_rate)-1

fig, ax = plt.subplots(2,3,sharex=True, sharey=True)
for m, state in enumerate(initial_states):
    for i, num_q in enumerate(num_q_vec):
        ax[m%2,m//2].plot(p_swap_vec, n_mat[:, i,m], 'o', label=f'{num_q} qubits')
    ax[m%2,m//2].set_title(f'teleportation of {state}')
    ax[m%2,m//2].set_xscale('log')
    if m%2:
        ax[m%2,m//2].set_xlabel('C-Z error')
    if (m // 2)-1:
        ax[m%2,m//2].set_ylabel('vector length')
    ax[m%2,m//2].grid(linestyle='--', linewidth=0.2)
fig.suptitle('long rang teleportation with post-processing correction')

plt.legend()
plt.show()





