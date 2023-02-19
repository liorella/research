
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
    circ.append("DEPOLARIZE1", qubits, p/10)
    circ.append('TICK')
    circ.append('H', qubits[1::2])
    circ.append("DEPOLARIZE1", qubits[1::2], p/10)
    circ.append('TICK')
    circ.append('CX', qubits[1:])
    circ.append("DEPOLARIZE2", qubits[1:], p)
    circ.append('TICK')
    circ.append('CX', qubits[:-1])
    circ.append("DEPOLARIZE2", qubits[:-1], p)
    circ.append('TICK')
    circ.append('H', qubits[::2][:-1])
    circ.append("DEPOLARIZE1", qubits[::2][:-1], p/10)
    circ.append('TICK')
    circ.append("DEPOLARIZE1", qubits[:-1], p)
    circ.append('TICK')
    circ.append('M', qubits[:-1])
    return circ

def target_correction(length, p, correction="X"):
    circ = stim.Circuit()
    target_qubit=length*2
    if correction=="X":
        circ.append('X', target_qubit)
    elif correction=="Z":
        circ.append('Z', target_qubit)
    circ.append("DEPOLARIZE1", target_qubit, p/10)
    circ.append('TICK')
    return circ

def measure_target(length, p, basis="Z"):
    circ = stim.Circuit()
    target_qubit=length*2
    circ.append("DEPOLARIZE1", target_qubit, p)
    if basis=="Z":
        circ.append('M', target_qubit)
    elif basis == "X":
        circ.append('MX', target_qubit)
    elif basis == "Y":
        circ.append('MY', target_qubit)
    return circ


##
length_vec=[2, 3, 4, 5]
shots = 50000
p_vec = np.logspace(-3, -0.5, num=10)
fail_rate = np.zeros((len(p_vec), len(length_vec)))

for k, length in tqdm(enumerate(length_vec)):
    for j, p in enumerate(p_vec):
        circ=full_circuit(length, p,initial_state="Z")
        #
        success = 0
        for shot in range(shots):
            sim = stim.TableauSimulator()
            mes_results = do_and_get_measure_results(sim, segment=circ, qubits_to_return=range(length*2))
            if (sum(mes_results[1::2]) % 2):
                sim.do(target_correction(length, p, correction="X"))
            if (sum(mes_results[::2]) % 2):
                sim.do(target_correction(length, p, correction="Z"))

            # checking teleportation
            target_state = do_and_get_measure_results(sim, segment=measure_target(length, p, basis="Z"), qubits_to_return=length*2)

            if target_state == 0:
                success += 1
        fail_rate[j,k]=(1-success / shots)
        # print(fail_rate)

np.savez('Tele_regular.npz', fail_rate=fail_rate,length_vec=length_vec, p_vec=p_vec)
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
plt.title('regular scheme')

plt.show()
##
fig, ax = plt.subplots(2,1)
ax[0].plot(p_vec, fail_rate[:,0], 'o', label='regular')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title('length 2')

ax[0].set(xlabel='physical error', ylabel='transmission error')
ax[0].grid(linestyle='--', linewidth=0.2)

NoRounds = np.load('Tele_noRounds_length2.npz')
ax[0].plot(NoRounds.f.p_vec, NoRounds.f.success_rate, 'o', label='surface 0 rounds')
ax[0].legend()
error_prob_reduction=(1-NoRounds.f.success_rate.T/fail_rate[:,0])*100

ax[1].plot(p_vec, error_prob_reduction.T, 'o', label='regular')
ax[1].set_xscale('log')
ax[1].set(xlabel='physical error', ylabel='reduction in teleportation error probability[%]')
ax[1].grid(linestyle='--', linewidth=0.2)
plt.show()
##



