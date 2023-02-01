import numpy as np
import stim
import matplotlib.pyplot as plt
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from simulate_qec_rounds_stim import gen_feedback_circuit
from simulate_qec_rounds_stim import gen_syndrome_circuit
from simulate_qec_rounds_stim import gen_anc_parity_circuit
from simulate_qec_rounds_stim import analyze_delta

from tqdm import tqdm
from numpy.linalg import inv

loading = False
saving = True
##
if loading:
    loaded_data = np.load('FT_adp_3nc.npz')
    exp_results = loaded_data.f.exp_results
    distance_vec = loaded_data.f.distance_vec
    p_vec = loaded_data.f.p_vec
    exp_rounds = loaded_data.f.exp_rounds
else:
    shots = 20000
    p_vec = np.logspace(-2.5, -0.7, num=15)
    distance_vec = range(3, 11, 2)
    exp_results = np.zeros((len(p_vec), len(distance_vec)))
    exp_rounds = np.zeros((len(p_vec), len(distance_vec)))
    for k, distance in (enumerate(distance_vec)):
        anc1 = range(4 * (distance - 1) + 1)[1::4]
        anc2 = range(4 * (distance - 1) + 1)[3::4]
        anc3 = range(4 * (distance - 1) + 1)[2::4]
        active_ancillas = np.vstack((anc1, anc2)).T.flatten()
        ancillas_to_correct_in_parity = np.vstack((anc3, anc2)).T.flatten()
        data_qubits = (range(4 * (distance - 1) + 1))[::4]
        matching_matrix = np.zeros((distance - 1, distance), dtype=np.uint8)
        for r in range(distance - 1):
            matching_matrix[r, r] = 1
            matching_matrix[r, r + 1] = 1
        synd2data = matching_matrix.T @ inv(matching_matrix @ matching_matrix.T % 2)
        t = np.floor((distance - 1) / 2)
        for j, p in tqdm(enumerate(p_vec)):
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
                            code_task='shor_style_3anc_syndrome_extraction',  # looks ok
                            distance=distance,
                            rounds=rounds,
                            t1_t2_depolarization=False,
                            meas_induced_dephasing_enhancement=False,
                            params=cparams
                            )
            segment2 = gen_syndrome_circuit(data_qubits, anc1, anc2, context)
            segment3 = gen_anc_parity_circuit(anc1, anc2, anc3, context)
            # running the exp.
            success = 0
            rounds_count = np.ones(shots)
            for shot in range(shots):
                reset_indicator = np.zeros(len(active_ancillas), dtype=np.uint8)
                sim = stim.TableauSimulator() #neccesary to add components into the circuit
                success_counter = 0
                fail_counter = 0
                rep_circ_iter = to_measure_segments(circuit)  # breaking down the circuit into segments, divided by measurement events
                delta=[]
                results_record = np.zeros((0,distance-1), dtype=np.uint8)
                end_condition = False
                for i, segment in enumerate(rep_circ_iter):
                    # break
                    if (i < 2) and (i % 2 ==0):
                        sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
                        parity_ancilla_mes = do_and_get_measure_results(sim, segment, anc3)  # simulate the segment
                        anc_parity_reset_indicator = np.vstack((parity_ancilla_mes, parity_ancilla_mes)).T.flatten()
                        sim.do(gen_feedback_circuit(anc_parity_reset_indicator, ancillas_to_correct_in_parity, context)) # gives an X gate for the ancilla the is to reset and append errors
                    elif i < 2:
                        ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
                        results = (np.diff(ancilla_mes)%2)[::2]
                        reset_indicator=ancilla_mes
                        results_record = np.concatenate((results_record, [results]))
                    elif i == 2:
                        while not(end_condition):
                            sim.do(gen_feedback_circuit(reset_indicator, active_ancillas,
                                                        context))  # gives an X gate for the ancilla the is to reset and append errors
                            parity_ancilla_mes = do_and_get_measure_results(sim, segment3, anc3)  # simulate the segment
                            anc_parity_reset_indicator = np.vstack((parity_ancilla_mes, parity_ancilla_mes)).T.flatten()
                            sim.do(gen_feedback_circuit(anc_parity_reset_indicator, ancillas_to_correct_in_parity,
                                                        context))  # gives an X gate for the ancilla the is to reset and append errors
                            rounds_count[shot]+=1
                            # print(fail_counter)
                            ancilla_mes = do_and_get_measure_results(sim, segment2, active_ancillas)
                            results = (np.diff(ancilla_mes) % 2)[::2]
                            results_record = np.concatenate((results_record, [results]))
                            delta.append(int((results_record[-2, :] != results).any()))
                            reset_indicator = ancilla_mes
                            if distance == 3:
                                if delta[-1] == 0:
                                    data_to_flip = (synd2data @ results) % 2
                                    if np.sum(data_to_flip) > t:
                                        data_to_flip = (data_to_flip + 1) % 2
                                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                                    end_condition = True
                                elif i > 1:
                                    if delta[-1] == delta[-2]:
                                        data_to_flip = (synd2data @ results) % 2
                                        if np.sum(data_to_flip) > t:
                                            data_to_flip = (data_to_flip + 1) % 2
                                        sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                                        end_condition = True
                            else:
                                [cond, locs, sum11]=analyze_delta(delta)
                                if sum11 == t:
                                    data_to_flip = (synd2data @ results) % 2
                                    if np.sum(data_to_flip) > t:
                                        data_to_flip = (data_to_flip + 1) % 2
                                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                                    end_condition = True
                                else:
                                    for k in range(len(cond)):
                                        if cond[k] > (t-1):
                                            data_to_flip = (synd2data @ results_record[locs[k],:]) % 2
                                            if np.sum(data_to_flip) > t:
                                                data_to_flip = (data_to_flip + 1) % 2
                                            sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                                            end_condition = True
                                            break
                        results_data = do_and_get_measure_results(sim, segment, data_qubits)
                        logical_state = np.around(np.sum(results_data)/context.distance, 0)
                        if logical_state == 0:
                            success += 1
            exp_results[j,k]=1-success/shots
            exp_rounds[j,k]=np.mean(rounds_count)
        print(exp_results[:,k])
    if saving:
        np.savez('FT_adp_3nc.npz', exp_results=exp_results, exp_rounds=exp_rounds, distance_vec=distance_vec, p_vec=p_vec)
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

plt.show()

plt.savefig('ShorFT_adp_3anc.svg')
##
fig2, ax2 = plt.subplots()
for i, distance in enumerate(distance_vec):
    ax2.plot(p_vec, exp_rounds[:,i], 'o', label=f'distance={distance}')

plt.xscale('log')
plt.legend()
plt.xlabel('physical error')
plt.ylabel('number of rounds until conversion')
plt.grid(linestyle='--', linewidth=0.2)

plt.show()

plt.savefig('ShorFT_adp_3anc_rounds.svg')

