
import numpy as np

import stim
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
import matplotlib.pyplot as plt
from tqdm import tqdm


## building circuit

def full_circuit(length, p,initial_state="Z"):
    circ = stim.Circuit()
    for i in range(2*length+1):
        circ.append('QUBIT_COORDS', [i], (i, i % 2))
    qubits = range(length*2+1)
    circ.append('R', qubits)
    if initial_state == "X":
        circ.append('RX', 0)
    elif initial_state == "Y":
        circ.append('RY', 0)
    for i in range(2*length):
        circ.append('H', i+1)
        circ.append("DEPOLARIZE1", [i+1], p/10)
        circ.append('TICK')
        circ.append('CZ', [i, i+1])
        circ.append("DEPOLARIZE2", qubits[1:], p)
        circ.append('TICK')
        circ.append('H', [i, i+1])
        circ.append("DEPOLARIZE1", [i, i+1], p/10)
        circ.append('TICK')
        circ.append('CZ', [i, i+1])
        circ.append("DEPOLARIZE2", qubits[1:], p)
        circ.append('TICK')
        circ.append('H', [i, i+1])
        circ.append("DEPOLARIZE1", [i, i+1], p/10)
        circ.append('TICK')
        circ.append('CZ', [i, i+1])
        circ.append("DEPOLARIZE2", qubits[1:], p)
        circ.append('TICK')
        circ.append('H', i+1)
        circ.append("DEPOLARIZE1", i+1, p/10)
        circ.append('TICK')
    return circ


def append_measure_target(circ, length, p, basis="Z"):
    target_qubit=length*2
    circ.append("DEPOLARIZE1", target_qubit, p)
    if basis=="Z":
        circ.append('M', target_qubit)
    elif basis == "X":
        circ.append('MX', target_qubit)
    elif basis == "Y":
        circ.append('MY', target_qubit)
    circ.append("DETECTOR", [stim.target_rec(-1)])

##
length_vec=[2, 3, 4, 5]
shots = 50000
p_vec = np.logspace(-3, -0.5, num=10)
fail_rate = np.zeros((len(p_vec), len(length_vec)))

for k, length in tqdm(enumerate(length_vec)):
    for j, p in enumerate(p_vec):
        circ=full_circuit(length, p,initial_state="Z")
        append_measure_target(circ, length, p, basis="Z")
        #
        success = 0
        sampler = circ.compile_detector_sampler()
        results = sampler.sample(shots=shots)
        fail_rate[j, k] = sum(results) / shots

        # print(fail_rate)

np.savez('Tele_swap.npz', fail_rate=fail_rate,length_vec=length_vec, p_vec=p_vec)
##
fig, ax = plt.subplots()
for i, length in enumerate(length_vec):
    ax.plot(p_vec, fail_rate[:,i], 'o', label=f'length={length}')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('physical error')
plt.ylabel('teleportation error')
plt.grid(linestyle='--', linewidth=0.2)
plt.title('teleportation via swap')

plt.show()
##
fig, ax = plt.subplots(2,1)
ax[0].plot(p_vec, fail_rate[:,0], 'o', label='teleportation via swap')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title('length 2')

ax[0].set(xlabel='physical error', ylabel='transmission error')
ax[0].grid(linestyle='--', linewidth=0.2)

NoRounds = np.load('Tele_noRounds_length2.npz')
regular = np.load('Tele_regular.npz')


ax[0].plot(NoRounds.f.p_vec, NoRounds.f.success_rate, 'o', label='surface 0 rounds')
ax[0].plot(regular.f.p_vec, regular.f.fail_rate[:,0], 'o', label='regular scheme')

ax[0].legend()

error_prob_reduction_1=(1-NoRounds.f.success_rate.T/fail_rate[:,0])*100
error_prob_reduction_2=(1-NoRounds.f.success_rate.T/regular.f.fail_rate[:,0])*100

ax[1].plot(p_vec, error_prob_reduction_1.T, 'o', label='surface vs swap')
ax[1].plot(p_vec, error_prob_reduction_2.T, 'o', label='surface vs regular')
ax[1].set_xscale('log')
ax[1].set(xlabel='physical error', ylabel='reduction in teleportation error probability[%]')
ax[1].grid(linestyle='--', linewidth=0.2)
ax[1].legend()

plt.show()
##



