## The simplest implementation of a repetition code using stim.
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from simulate_qec_rounds_stim import gen_feedback_circuit
from tqdm import tqdm

loading = True
saving = False
##
if loading:
    loaded_data = np.load('NonFT_Shor_repetition_stim.npz')
    exp_results = loaded_data.f.exp_results
    distance_vec = loaded_data.f.distance_vec
    p_vec = loaded_data.f.p_vec
else:
    shots = 20000
    p_vec = np.logspace(-2.5, -0.7, num=15)
    distance_vec = range(3, 11, 2)
    exp_results = np.zeros((len(p_vec),len(distance_vec)))

    for k, distance in (enumerate(distance_vec)):
        print(distance)
        rounds = distance
        num_rounds = rounds
        active_ancillas = np.vstack((range(3 * (distance - 1) + 1)[1::3], range(3 * (distance - 1) + 1)[2::3])).T.flatten()
        data_qubits = (range(3 * (distance - 1) + 1))[::3]
        matching_matrix = np.zeros((distance - 1, distance), dtype=np.uint8)
        for r in range(distance - 1):
            matching_matrix[r, r] = 1
            matching_matrix[r, r + 1] = 1
        m = pymatching.Matching(matching_matrix, repetitions=rounds + 1)

        for j, p in tqdm(enumerate(p_vec)):
            cparams = CircuitParams(t1=0,  # if t1=0 than use the single probability
                                    t2=0,
                                    single_qubit_gate_duration=20,
                                    two_qubit_gate_duration=20,
                                    single_qubit_depolarization_rate=p,
                                    two_qubit_depolarization_rate=p,
                                    meas_duration=550,
                                    reset_duration=0,
                                    reset_latency=40,
                                    meas_induced_dephasing_enhancement=0)
            circuit, context , _ = generate_scheduled(
                            code_task='shor_style_syndrome_extraction',  # looks ok
                            distance=distance,
                            rounds=rounds,
                            t1_t2_depolarization=False,
                            meas_induced_dephasing_enhancement=False,
                            params=cparams
                            ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p


        # running the exp.
            success = 0
            logical_counter = 0 #checks if after the decoding there is a difference between the bits

            for shot in range(shots):
                rep_circ_iter = to_measure_segments(circuit) #breaking down the circuit into segments, divided by measurement events
                reset_indicator = np.zeros(len(active_ancillas), dtype=np.uint8)
                results_record = np.zeros((num_rounds + 1, distance-1), dtype=np.uint8) # a matrix that will eventually reach the decoder
                sim = stim.TableauSimulator() #neccesary to add components into the circuit

                for i, segment in enumerate(rep_circ_iter):
                    if i < num_rounds:
                        sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
                        ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
                        results = (np.diff(ancilla_mes)%2)[::2]
                        results_record[i, :] = results
                        reset_indicator = ancilla_mes
                    elif i == num_rounds:
                        results_data = do_and_get_measure_results(sim, segment, data_qubits)
                        data_parity = matching_matrix @ results_data % 2
                        results_record[i, :] = data_parity
                        to_decode = np.diff(np.hstack((np.zeros(distance-1, dtype=np.uint8)[:, np.newaxis], results_record.T))) % 2
                        frame = m.decode(to_decode)
                        corrected_state = (results_data + frame) % 2
                        logical_state = corrected_state[-1]
                        if logical_state == 0:
                            success += 1
            exp_results[j,k]=1-success/shots
    if saving:
        np.savez('NonFT_Shor_repetition_stim.npz', exp_results=exp_results, distance_vec=distance_vec, p_vec=p_vec)


## plotting
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

plt.savefig('ShorNonFT_repetition_stim.svg')
