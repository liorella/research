

import numpy as np
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled

num_q_vec=[3,5,7]
shots = 50000
initial_states=["Z","mZ","X","mX","Y","mY"]




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



## via swap
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

## long range post-processing


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

