from math import exp, sqrt
import stim
import numpy as np
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from matplotlib import pyplot as plt
import stimcirq
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

num_q_vec=[3,5,7]
shots = 50000
initial_states=["Z","mZ","X","mX","Y","mY"]

## params
Readout_fidelity_ground={'q4_8': 0.99466, 'q4_7': 0.99706, 'q5_7': 0.99512, 'q5_6': 0.99606, 'q6_6': 0.99588, 'q6_5': 0.99696}
Readout_fidelity_excited={'q4_8': 0.98062, 'q4_7': 0.96644, 'q5_7': 0.97386, 'q5_6': 0.9729, 'q6_6': 0.98488, 'q6_5': 0.98302}
Single_qubit_error= {'q3_8': 0.000651, 'q4_9': 0.000698,'q4_8': 0.001146, 'q4_7': 0.001351, 'q5_7': 0.000817, 'q5_6': 0.000862, 'q6_6': 0.0007, 'q6_5': 0.000664}
T1_all= {'q3_8': 13.46, 'q4_9': 13.36,'q4_8': 15.09, 'q4_7': 12.86, 'q5_7': 13.7, 'q5_6': 19.5, 'q6_6': 22.62, 'q6_5': 19.08}
XEB= {'q4_8-q3_8': 0.0055,'q4_8-q4_9': 0.0062,'q4_8-q4_7': 0.007,'q5_7-q4_7': 0.0073,'q5_6-q5_7': 0.008,'q6_6-q5_6': 0.0087, 'q6_6-q6_5': 0.0091}

## streight-forward circuit


single_qubit_depolarization_rates=list(Single_qubit_error.values())[1::] #the first qubit is the target qubit
Two_qubit_depolarization_rates=np.array(list(XEB.values())[1::])*1.25 #the first qubit is the target qubit
T1=np.array(list(T1_all.values())[1::])*1000
T2=np.array(T1)/2
readout_error=np.ones(7)
readout_error[1:] = readout_error[1:]-(np.array(list(Readout_fidelity_ground.values()))+np.array(list(Readout_fidelity_excited.values())))/2
readout_error[0] = np.average(readout_error[1:])
single_qubit_gate_duration=25
two_qubit_gate_duration=25
meas_duration=560 #meas_duratio+feedback delay


@dataclass
class SpecificCircuitParams:
    t1: list
    t2: list
    single_qubit_depolarization_rate: list
    two_qubit_depolarization_rate: list
    measurement_error: list
    meas_duration: float
    single_qubit_gate_duration: float
    two_qubit_gate_duration: float
    reset_duration: float
    reset_latency: float


cparams = SpecificCircuitParams(t1=T1,
                t2=T2,
                single_qubit_depolarization_rate=single_qubit_depolarization_rates,
                two_qubit_depolarization_rate=Two_qubit_depolarization_rates,
                measurement_error=readout_error,
                meas_duration=meas_duration,
                reset_duration=0,
                reset_latency=0,
                single_qubit_gate_duration=single_qubit_gate_duration,
                two_qubit_gate_duration=two_qubit_gate_duration,
                )
##
def add_idle_errors(circ:stim.Circuit, idle_qubits, cparams:SpecificCircuitParams, duration):
    for idle_qubit in idle_qubits:
        circ.append_operation('PAULI_CHANNEL_1',
                                  idle_qubit,
                              ((1 - exp(-duration / cparams.t1[idle_qubit])) / 4, (1 - exp(-duration / cparams.t1[idle_qubit])) / 4,
                              (1 - exp(-duration / cparams.t2[idle_qubit])) / 2 - (1 - exp(-duration / cparams.t1[idle_qubit])) / 4))

def add_depolarization_single_qubit_errors(circ:stim.Circuit, qubits, cparams:SpecificCircuitParams):
    for qubit in qubits:
        circ.append("DEPOLARIZE1", qubit, cparams.single_qubit_depolarization_rate[qubit])

def add_depolarization_two_qubit_errors(circ:stim.Circuit, qubit_pairs, cparams:SpecificCircuitParams):
    pair_list=np.resize(qubit_pairs,(int(0.5*len(qubit_pairs)),2))
    for qubit_pair in pair_list:
        circ.append("DEPOLARIZE2", qubit_pair, cparams.two_qubit_depolarization_rate[qubit_pair[0]])

def add_measurement_error(circ:stim.Circuit, qubits, cparams:SpecificCircuitParams):
    for qubit in qubits:
        circ.append("X_ERROR", qubit, cparams.measurement_error[qubit])


def teleportation_circuit(circ,distance, initial_state, cparams:SpecificCircuitParams, epoch):
    qubits = range(distance)
    if epoch == 0:
        circ.append('R', qubits)
        if initial_state == "X":
            circ.append('H', distance-1)
            add_depolarization_single_qubit_errors(circ, [distance-1], cparams)
        elif initial_state == "Y":
            circ.append('H_YZ', distance-1)
            add_depolarization_single_qubit_errors(circ, [distance-1], cparams)
        elif initial_state == "mZ":
            circ.append('X', distance-1)
            add_depolarization_single_qubit_errors(circ, [distance-1], cparams)
        elif initial_state == "mX":
            circ.append('H', distance-1)
            circ.append('Z', distance-1)
            add_depolarization_single_qubit_errors(circ, [distance-1], cparams)
        elif initial_state == "mY":
            circ.append('H_YZ', distance-1)
            circ.append('Z', distance-1)
            add_depolarization_single_qubit_errors(circ, [distance-1], cparams)
        add_idle_errors(circ, range(distance-1), cparams, cparams.single_qubit_gate_duration)
    elif epoch == 1:
        circ.append('H', qubits)
        add_depolarization_single_qubit_errors(circ, qubits, cparams)
    elif epoch == 2:
        circ.append('CZ', qubits[1:])
        add_depolarization_two_qubit_errors(circ, qubits[1:], cparams)
        add_idle_errors(circ, [qubits[0]], cparams, cparams.single_qubit_gate_duration)
    elif epoch == 3:
        circ.append('CZ', qubits[:-1])
        add_depolarization_two_qubit_errors(circ, qubits[:-1], cparams)
        add_idle_errors(circ, [qubits[-1]], cparams, cparams.single_qubit_gate_duration)
    elif epoch == 4:
        circ.append('H', qubits)
        add_depolarization_single_qubit_errors(circ, qubits, cparams)
    elif epoch == 5:
        add_measurement_error(circ, qubits[1:], cparams)
        circ.append('M', qubits[1:])
        add_idle_errors(circ, [qubits[0]], cparams, cparams.meas_duration)
        if distance == 3:
            circ.append("DETECTOR", [stim.target_rec(-1)])
            circ.append("DETECTOR", [stim.target_rec(-2)])
        elif distance == 5:
            circ.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
            circ.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4)])
        else:
            circ.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3), stim.target_rec(-5)])
            circ.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4), stim.target_rec(-6)])


def measure_target(circ, single_qubit_depolarization_rate, before_measure_flip_probability, initial_state):
    target_qubit = 0
    if initial_state == "mZ":
        circ.append('X', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit,single_qubit_depolarization_rate)
    elif initial_state == "X":
        circ.append('H', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, single_qubit_depolarization_rate)
    elif initial_state == "Y":
        circ.append('H_YZ', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, single_qubit_depolarization_rate)
    elif initial_state == "mX":
        circ.append('Z', target_qubit)
        circ.append('H', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, single_qubit_depolarization_rate)
    elif initial_state == "mY":
        circ.append('Z', target_qubit)
        circ.append('H_YZ', target_qubit)
        circ.append("DEPOLARIZE1", target_qubit, single_qubit_depolarization_rate)
    circ.append("X_ERROR", target_qubit, before_measure_flip_probability)
    circ.append('M', target_qubit)
    circ.append("DETECTOR", [stim.target_rec(-1)])




## via post-processing

n_mat = np.zeros((len(num_q_vec),len(initial_states)))
fidelity = np.zeros((len(num_q_vec),len(initial_states)))
for m, state in enumerate(initial_states):
    for k, distance in enumerate(num_q_vec):
        circ = stim.Circuit()
        for epoch in range(6):
            teleportation_circuit(circ, distance, state, cparams, epoch)
        measure_target(circ, cparams.single_qubit_depolarization_rate[0], cparams.measurement_error[0], state)
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
        fidelity[k,m] = 1-fail_rate

print(n_mat)
print(fidelity)

exp_mat=[[0.925, 0.9, 0.88],
         [0.935, 0.925, 0.90],
         [0.915, 0.885, 0.85]]


##

exp_mat=[[0.93, 0.9, 0.88],
         [0.935, 0.925, 0.90],
         [0.915, 0.885, 0.85]]
fidelity_T2isT1 = np.transpose([[0.9361,  0.93614, 0.94248, 0.94276, 0.9228,  0.92118],
 [0.91424, 0.91456, 0.9185,  0.91798, 0.88584, 0.88558],
 [0.8955,  0.89306, 0.90238, 0.89808, 0.8609,  0.85966]])

fidelity_T2is2T1=np.transpose([[0.93486, 0.934,   0.92482, 0.92478, 0.90498, 0.90246],
 [0.91042, 0.90988, 0.9026,  0.89924, 0.86884, 0.86962],
 [0.8908,  0.89242, 0.883,   0.88202, 0.84304, 0.84388]])

fig, ax = plt.subplots()
num_q_vec=np.array([3,5,7])
bar_width=0.15
shift=0.2
avg_fidelity_sim=(fidelity_T2isT1+fidelity_T2is2T1)/2
avg_fidelity_sim_variance=(fidelity_T2isT1-fidelity_T2is2T1)


for m, state in enumerate(initial_states):
    if m==0:
        plt.bar(num_q_vec+shift, exp_mat[0], bar_width,color='#8b008b')
        plt.errorbar(num_q_vec+shift, avg_fidelity_sim[m],avg_fidelity_sim_variance[m],fmt="o", c='#ff69B4', markersize=5, capsize=3, ecolor='black')
    elif m==2:
        plt.bar(num_q_vec-shift, exp_mat[1],bar_width,color='#008000')
        plt.errorbar(num_q_vec-shift, avg_fidelity_sim[m],avg_fidelity_sim_variance[m],fmt="o", c='#00ff00', markersize=5, capsize=3, ecolor='black')
    elif m==4:
        plt.bar(num_q_vec, exp_mat[2],bar_width,color='#FF0000')
        plt.errorbar(num_q_vec, avg_fidelity_sim[m],avg_fidelity_sim_variance[m],fmt="o", c='#ffc000', markersize=5, capsize=3, ecolor='black')



plt.show()
plt.xlabel('# of qubits')
plt.ylim((0.6,1))
plt.xlim((1,9))
plt.xticks([3,5,7])
plt.ylabel('Fidelity')

presentation_bg_color = np.array([234, 242, 245]) / 255
plt.rcParams['axes.facecolor'] = presentation_bg_color
plt.rcParams['figure.facecolor'] = presentation_bg_color
##

fig, ax = plt.subplots()
for m, state in enumerate(initial_states):
    if m==0:
        ax.scatter(num_q_vec, fidelity[:, m], label=f'initial state ={state}, sim', marker='o',c='#1f77b4')
        ax.scatter(num_q_vec, exp_mat[0], label=f'initial state ={state}, exp', marker='x', c='#1f77b4')
    elif m==2:
        ax.scatter(num_q_vec, fidelity[:, m], label=f'initial state ={state}, sim',marker='o',c='#ff7f0e')
        ax.scatter(num_q_vec, exp_mat[1], label=f'initial state ={state}, exp', marker='x', c='#ff7f0e')
    elif m==4:
        ax.scatter(num_q_vec, fidelity[:, m], label=f'initial state ={state}, sim',marker='o',c='#2ca02c')
        ax.scatter(num_q_vec, exp_mat[2], label=f'initial state ={state}, exp', marker='x', c='#2ca02c')

plt.title('long-range teleportation with post-processing')
plt.xlabel('# of qubits')
plt.ylabel('Fidelity')
plt.ylim((0.7,1))
plt.legend()
plt.show()

