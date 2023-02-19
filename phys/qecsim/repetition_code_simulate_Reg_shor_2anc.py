## The simplest implementation of a repetition code using stim.
import numpy as np

import stim
import matplotlib.pyplot as plt
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from simulate_qec_rounds_stim import gen_feedback_circuit
from simulate_qec_rounds_stim import gen_syndrome_circuit
from numpy.linalg import inv
from tqdm import tqdm

loading = True
saving = False
##
if loading:
    loaded_data = np.load('FT_Reg_2nc.npz')
    exp_results = loaded_data.f.exp_results
    distance_vec = loaded_data.f.distance_vec
    p_vec = loaded_data.f.p_vec
    exp_rounds = loaded_data.f.exp_rounds
else:
    shots = 50000
    p_vec = np.logspace(-3.2, -0.5, num=20)
    distance_vec = range(3, 11, 2)
    exp_results = np.zeros((len(p_vec), len(distance_vec)))
    exp_rounds = np.zeros((len(p_vec), len(distance_vec)))

    for k, distance in tqdm(enumerate(distance_vec)):
        anc1 = range(3 * (distance - 1) + 1)[1::3]
        anc2 = range(3 * (distance - 1) + 1)[2::3]
        active_ancillas = np.vstack((anc1, anc2)).T.flatten()
        data_qubits = (range(3 * (distance - 1) + 1))[::3]
        matching_matrix = np.zeros((distance - 1, distance), dtype=np.uint8)
        for r in range(distance - 1):
            matching_matrix[r, r] = 1
            matching_matrix[r, r + 1] = 1
        synd2data = matching_matrix.T @ inv(matching_matrix @ matching_matrix.T % 2)
        t = np.floor((distance - 1) / 2)
        for j, p in (enumerate(p_vec)):
            cparams = CircuitParams(t1=0, #if t1=0 than use the single probability
                            t2=0,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=20,
                            single_qubit_depolarization_rate=p,
                            two_qubit_depolarization_rate=p,
                            meas_duration=550,
                            reset_duration=0,
                            reset_latency=40,
                            meas_induced_dephasing_enhancement=3)

            circuit, context , _ = generate_scheduled(
                            code_task='shor_style_syndrome_extraction',  # looks ok
                            distance=distance,
                            rounds=1,
                            t1_t2_depolarization=False,
                            meas_induced_dephasing_enhancement=False,
                            params=cparams
                            ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p
            segment2 = gen_syndrome_circuit(data_qubits, anc1, anc2, context)
                # running the exp.
            success = 0
            rounds_count = np.ones(shots)

            for shot in range(shots):
                reset_indicator = np.zeros(len(active_ancillas), dtype=np.uint8)
                sim = stim.TableauSimulator() #neccesary to add components into the circuit
                success_counter = 0
                fail_counter = 0
                rep_circ_iter = to_measure_segments(
                    circuit)  # breaking down the circuit into segments, divided by measurement events
                results_prev = np.zeros(distance-1)
                for i, segment in enumerate(rep_circ_iter):
                    # break
                    if i < 1:
                        sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
                        ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
                        results = (np.diff(ancilla_mes)%2)[::2]
                        if (results==results_prev).all():
                            success_counter+=1
                            if success_counter == t+1:
                                success_counter = 0
                                fail_counter = 0
                                data_to_flip = (synd2data @ results) % 2
                                if np.sum(data_to_flip)>t:
                                    data_to_flip = (data_to_flip+1) % 2
                                sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                        else:
                            fail_counter+=1
                            success_counter=0
                            if fail_counter == (t+1)**2:
                                fail_counter = 0
                                data_to_flip = (synd2data @ results).astype(np.uint8)
                                if np.sum(data_to_flip)>t:
                                    data_to_flip = (data_to_flip+1) % 2
                                sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                        reset_indicator = ancilla_mes
                        results_prev = results
                    elif i == 1:
                     #  break
                        while 0 < success_counter < t+1 or 0 < fail_counter < (t+1)**2:
                            sim.do(gen_feedback_circuit(reset_indicator, active_ancillas,
                                                        context))  # gives an X gate for the ancilla the is to reset and append errors
                            rounds_count[shot]+=1
                            # print(fail_counter)
                            ancilla_mes = do_and_get_measure_results(sim, segment2, active_ancillas)
                            results = (np.diff(ancilla_mes) % 2)[::2]
                            if (results == results_prev).all():
                                success_counter += 1
                                if success_counter == t + 1:
                                    fail_counter = 0
                                    data_to_flip = (synd2data @ results) % 2
                                    if np.sum(data_to_flip) > t:
                                        data_to_flip = (data_to_flip + 1) % 2
                                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                            else:
                                fail_counter += 1
                                success_counter = 0
                                if fail_counter == (t + 1) ** 2:
                                    data_to_flip = (synd2data @ results).astype(np.uint8)
                                    if np.sum(data_to_flip) > t:
                                        data_to_flip = (data_to_flip + 1) % 2
                                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                            results_prev = results
                            reset_indicator = ancilla_mes
                        results_data = do_and_get_measure_results(sim, segment, data_qubits)
                        logical_state = np.around(np.sum(results_data)/context.distance, 0)
                        if logical_state == 0:
                            success += 1
            exp_results[j,k]=1-success/shots
            exp_rounds[j,k]=np.mean(rounds_count)
        print(exp_results[:,k])
    if saving:
        np.savez('FT_Reg_2nc.npz', exp_results=exp_results, exp_rounds=exp_rounds, distance_vec=distance_vec, p_vec=p_vec)

## plotting
print(exp_results)
fig, ax = plt.subplots()
for i, distance in enumerate(distance_vec):
    ax.plot(p_vec, exp_results[:,i], 'o', label=f'distance={distance}')
ax.plot(p_vec, p_vec, label=f'single qubit')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('physical error')
plt.ylabel('logical error')
plt.grid(linestyle='--', linewidth=0.2)
plt.xlim([10**(-2.5),0.3])
plt.ylim([0.5E-3, 0.6])

plt.show()

plt.savefig('ShorFT_repetition_2anc.svg')
##
fig2, ax2 = plt.subplots()
for i, distance in enumerate(distance_vec):
    ax2.plot(p_vec, exp_rounds[:,i], 'o', label=f'distance={distance}')

plt.xscale('log')
plt.legend()
plt.xlabel('physical error')
plt.ylabel('number of rounds until conversion')
plt.grid(linestyle='--', linewidth=0.2)
plt.xlim([10**(-2.5),0.3])

plt.show()

plt.savefig('ShorFT_repetition_2anc_rounds.svg')
