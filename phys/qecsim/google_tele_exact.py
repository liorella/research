

import numpy as np
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from matplotlib import pyplot as plt

num_q_vec=[3,5,7]
shots = 50000
initial_states=["Z","mZ","X","mX","Y","mY"]


## via swap
cparams = CircuitParams(t1=6e3,
                t2=6e3,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=40,
                single_qubit_depolarization_rate=1.2e-3,
                two_qubit_depolarization_rate=7.5e-3,
                meas_duration=350,
                reset_duration=0,
                reset_latency=0,
                measurement_error=2E-2,
                meas_induced_dephasing_enhancement=1)



n_mat = np.zeros((len(num_q_vec),len(initial_states)))

for m, state in enumerate(initial_states):
    for k, distance in enumerate(num_q_vec):
        circ, cont, _ = generate_scheduled(
            code_task='teleportation_via_swap',
            distance=distance,
            rounds=0,
            params=cparams,
            initial_state=state
        )
        success = 0
        sampler = circ.compile_detector_sampler()
        results = sampler.sample(shots=shots)
        fail_rate = sum(results) / shots
        n_mat[k,m]=2*(1-fail_rate)-1
print(n_mat)
##

exp_mat=[[0.91,0.87,0.8],
         [0.82, 0.83, 0.81],
         [0.85, 0.77, 0.66]]

fig, ax = plt.subplots()
for m, state in enumerate(initial_states):
    if m==0:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#1f77b4')
        ax.scatter(num_q_vec, exp_mat[0], label=f'initial state ={state}, exp', marker='x', c='#1f77b4')
    elif m==2:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#ff7f0e')
        ax.scatter(num_q_vec, exp_mat[1], label=f'initial state ={state}, exp', marker='x', c='#ff7f0e')
    elif m==4:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#2ca02c')
        ax.scatter(num_q_vec, exp_mat[2], label=f'initial state ={state}, exp', marker='x', c='#2ca02c')

plt.title('teleportation via SWAP')
plt.xlabel('# of qubits')
plt.ylabel('Bloch-sphere vector length')
plt.ylim((0.5,1))

plt.legend()
plt.show()
## long range post-processing

cparams = CircuitParams(t1=6e3,
                t2=6e3,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=40,
                single_qubit_depolarization_rate=1.2e-3,
                two_qubit_depolarization_rate=7.5e-3,
                meas_duration=350,
                reset_duration=0,
                reset_latency=0,
                measurement_error=2E-2,
                meas_induced_dephasing_enhancement=1)

n_mat = np.zeros((len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, distance in enumerate(num_q_vec):
        circ, cont, _ = generate_scheduled(
            code_task='teleportation_post_processing',
            distance=distance,
            rounds=0,
            params=cparams,
            initial_state=state
        )
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

##
exp_mat=[[0.82, 0.77, 0.7],
         [0.88, 0.76, 0.7],
         [0.82, 0.71, 0.62]]

fig, ax = plt.subplots()
for m, state in enumerate(initial_states):
    if m==0:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim', marker='o',c='#1f77b4')
        ax.scatter(num_q_vec, exp_mat[0], label=f'initial state ={state}, exp', marker='x', c='#1f77b4')
    elif m==2:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#ff7f0e')
        ax.scatter(num_q_vec, exp_mat[1], label=f'initial state ={state}, exp', marker='x', c='#ff7f0e')
    elif m==4:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#2ca02c')
        ax.scatter(num_q_vec, exp_mat[2], label=f'initial state ={state}, exp', marker='x', c='#2ca02c')

plt.title('long-range teleportation with post-processing')
plt.xlabel('# of qubits')
plt.ylabel('Bloch-sphere vector length')
plt.ylim((0.5,1))

plt.legend()
plt.show()
## long range feedback

cparams = CircuitParams(t1=6e3,
                t2=6e3,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=40,
                single_qubit_depolarization_rate=1.2e-3,
                two_qubit_depolarization_rate=7.5e-3,
                meas_duration=700,
                reset_duration=0,
                reset_latency=0,
                measurement_error=2.1E-2,
                meas_induced_dephasing_enhancement=1)

n_mat = np.zeros((len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, distance in enumerate(num_q_vec):
        circ, cont, _ = generate_scheduled(
            code_task='teleportation_post_processing',
            distance=distance,
            rounds=0,
            params=cparams,
            initial_state=state
        )
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

##
exp_mat=[[0.82, 0.77, 0.71],
         [0.78, 0.7, 0.68],
         [0.74, 0.64, 0.55]]

fig, ax = plt.subplots()
for m, state in enumerate(initial_states):
    if m==0:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim', marker='o',c='#1f77b4')
        ax.scatter(num_q_vec, exp_mat[0], label=f'initial state ={state}, exp', marker='x', c='#1f77b4')
    elif m==2:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#ff7f0e')
        ax.scatter(num_q_vec, exp_mat[1], label=f'initial state ={state}, exp', marker='x', c='#ff7f0e')
    elif m==4:
        ax.scatter(num_q_vec, n_mat[:, m], label=f'initial state ={state}, sim',marker='o',c='#2ca02c')
        ax.scatter(num_q_vec, exp_mat[2], label=f'initial state ={state}, exp', marker='x', c='#2ca02c')

plt.title('long-range teleportation with feed-forward')
plt.xlabel('# of qubits')
plt.ylabel('Bloch-sphere vector length')
plt.ylim((0.5,1))

plt.legend()
plt.show()