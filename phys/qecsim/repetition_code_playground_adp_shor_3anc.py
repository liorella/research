## The simplest implementation of a repetition code using stim.
import numpy as np

import stim
import pymatching
import matplotlib.pyplot as plt
import networkx as nx

from stim_lib.run_feedback import measure_instructions

from qec_generator import CircuitParams
from stim_lib.error_context import StimErrorContext
from stim_lib.scheduled_circuit import generate_scheduled, get_pauli_probs, instruction_duration
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
from qec_generator import CircuitParams
from simulate_qec_rounds_stim import experiment_run
from stim_lib.scheduled_circuit import generate_scheduled
from simulate_qec_rounds_stim import gen_feedback_circuit
from simulate_qec_rounds_stim import gen_syndrome_circuit
from simulate_qec_rounds_stim import gen_anc_parity_circuit

from numpy.linalg import inv

from scipy.optimize import curve_fit
##

p=0.003

rounds=8
distance=3

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
                ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p
# The context include the matching matrix

## defining manually without touching the context

num_rounds = rounds
anc1 = range(4 * (distance - 1) + 1)[1::4]
anc2 = range(4 * (distance - 1) + 1)[3::4]
anc3 = range(4 * (distance - 1) + 1)[2::4]
active_ancillas = np.vstack((anc1, anc2)).T.flatten()
ancillas_to_correct_in_parity = np.vstack((anc3, anc2)).T.flatten()

data_qubits = (range(4 * (distance - 1) + 1))[::4]

matching_matrix=np.zeros((distance-1, distance), dtype=np.uint8)
for r in range(distance-1):
    matching_matrix[r, r] = 1
    matching_matrix[r, r+1] = 1

synd2data = matching_matrix.T @ inv(matching_matrix@matching_matrix.T % 2)

segment2 = gen_syndrome_circuit(data_qubits, anc1, anc2, context)

segment3 = gen_anc_parity_circuit(anc1, anc2, anc3, context)

t = np.floor((distance-1)/2)
##
def analyze_delta(delt: np.ndarray):
    # delt = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    gamma = []
    location_of_zeros = []
    ind_init=[]
    ind_end=[]
    ind_ones_init=[]
    ind_ones_end=[]
    sum11=0
    for i in range(len(delt)):
        if delt[i]==0:
            if i == 0:
                gamma.append(1)
                location_of_zeros.append(0)
                ind_init.append(0)
            else:
                if delt[i-1] == 1:
                    ind_init.append(i-1)
                    gamma.append(1)
                    location_of_zeros.append(i+1)
                    ind_ones_end.append(i-1)
                else:
                    gamma[-1] += 1
                if i == len(delt) - 1:
                    ind_end.append(i)
        else:
            if i > 0:
                if delt[i - 1] == 0:
                    ind_ones_init.append(i)
                    ind_end.append(i)
                if i==len(delt)-1:
                    ind_ones_end.append(i)
            else:
                ind_ones_init.append(0)
    if len(gamma)==0:
        sum11 += len(delt)-1
        gamma=[0]
        alpha=[0]
        beta=[0]
    else:
        alpha = np.zeros(len(gamma))
        beta = np.zeros(len(gamma))

        for j in range(len(ind_ones_end)):
            if (ind_ones_init[j] < ind_ones_end[j]):
                sum11+=ind_ones_end[j]-ind_ones_init[j]
            for i in range(len(gamma)):
                if (ind_ones_end[j] > ind_end[i]):
                    if (ind_ones_init[j] == ind_ones_end[j]):
                        beta[i]+=1
                    else:
                        beta[i] += ind_ones_end[j]-ind_ones_init[j]
                if (ind_ones_init[j] < ind_init[i]):
                    if (ind_ones_init[j] == ind_ones_end[j]):
                        alpha[i]+=1
                    else:
                        alpha[i] += ind_ones_end[j]-ind_ones_init[j]
        alpha[0]=0
        beta[-1]=0
    # print(delt)
    # print(alpha)
    # print(beta)
    # print(gamma)
    return alpha+beta+gamma, location_of_zeros, sum11


## running the exp.
success = 0
shots = 1000
rounds_count = rounds*np.ones(shots)

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
        if (i < num_rounds * 2) and (i % 2 ==0):
            sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
            parity_ancilla_mes = do_and_get_measure_results(sim, segment, anc3)  # simulate the segment
            anc_parity_reset_indicator = np.vstack((parity_ancilla_mes, parity_ancilla_mes)).T.flatten()
            sim.do(gen_feedback_circuit(anc_parity_reset_indicator, ancillas_to_correct_in_parity, context)) # gives an X gate for the ancilla the is to reset and append errors
        elif i < num_rounds * 2:
            ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
            results = (np.diff(ancilla_mes)%2)[::2]
            reset_indicator=ancilla_mes
            results_record = np.concatenate((results_record, [results]))
            if results_record.shape[0] > 1:
                delta.append(int((results_record[-2, :] != results).any()))
                end_condition = False
                if distance == 3:
                    if delta[-1] == 0:
                        data_to_flip = (synd2data @ results) % 2
                        if np.sum(data_to_flip) > t:
                            data_to_flip = (data_to_flip + 1) % 2
                        sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                        end_condition = True
                    elif i > 3:
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
                        delta=[]
                        results_record = np.zeros((0, distance - 1), dtype=np.uint8)
                        end_condition = True
                    else:
                        for k in range(len(locs)):
                            if cond[k] > (t-1):
                                data_to_flip = (synd2data @ results_record[locs[k],:]) % 2
                                if np.sum(data_to_flip) > t:
                                    data_to_flip = (data_to_flip + 1) % 2
                                sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                                delta = []
                                results_record = np.zeros((0, distance - 1), dtype=np.uint8)
                                end_condition = True
                                break

        elif i == num_rounds* 2:
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

log_error_prob = 1-success/shots
print(log_error_prob)

print(np.mean(rounds_count))

