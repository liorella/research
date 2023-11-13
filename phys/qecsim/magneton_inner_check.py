import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate.credentials import create_credentials
from qm import SimulationConfig
from configuration import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import stim
import numpy as np
# qmm = QuantumMachinesManager("tyler-263ed49e.dev.quantum-machines.co", port=443, credentials=create_credentials())
#qmm = QuantumMachinesManager(host='product-52ecaa43.dev.quantum-machines.co', port=443,credentials=create_credentials()) #cluser
qmm = QuantumMachinesManager(host="172.16.30.254", cluster_name='my_cluster_2') # OPX in research team room
## calculate simple error probabilities
relevant_detectors= np.array([0, 1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15])

simple_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=2,
    distance=3,
    after_clifford_depolarization= 0.3,
    before_round_data_depolarization=0,
    before_measure_flip_probability=0.3)
model=simple_circuit.detector_error_model()
probs=np.zeros(model.num_detectors)

for j in range(len(model)):
    if type(model[j])==stim._stim_sse2.DemInstruction and model[j].type=='error':
        for ind in range(len(model[j].targets_copy())):
            for i in range(model.num_detectors):
                if model[j].targets_copy()[ind] == stim.target_relative_detector_id(i):
                    probs[i] = probs[i]*(1-model[j].args_copy()[0])+(1-probs[i])*model[j].args_copy()[0]

rel_probs=np.zeros(len(relevant_detectors))
for i,ind in enumerate(relevant_detectors):
    rel_probs[i]=probs[ind]

print(rel_probs)


th = -0.0005
##
use_simulator = True
from qm.qua import Random
num_rounds=4
with program() as prog:
    generator = Random()
    probs=declare(fixed, value=rel_probs)
    probs_init = declare(fixed, value=rel_probs[0:4])
    probs_mid = declare(fixed, value=rel_probs[4:8])
    probs_fin = declare(fixed, value=rel_probs[8:])
    boo_vec = declare(int, size=4)
    boo_vec_final = declare(int, size=4)
    cond=declare(bool)
    boo = declare(int)
    boo_final = declare(int)
    i = declare(int)
    out = declare(fixed)
    assign(boo_final, 0)
    assign(boo, 0)
    with for_(i, 0, i < num_rounds, i+1):
        with if_(i == 0):
            for j in range(4):
                a = generator.rand_fixed()
                assign(cond, a<probs_init[j])
                assign(boo, boo + (Cast.to_int(cond) << j))
                b = generator.rand_fixed()
                assign(cond, b<probs_fin[j])
                assign(boo_final, boo_final + (Cast.to_int(cond) << j))
        with else_():
            for j in range(4):
                a = generator.rand_fixed()
                assign(cond, a<probs_init[j])
                assign(boo, boo + (Cast.to_int(cond) << j))
        align()
        with switch_(boo):
            with case_(0):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
            with case_(1):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
            with case_(2):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
            with case_(3):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
            with case_(4):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
            with case_(5):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
            with case_(6):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
            with case_(7):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
            with case_(8):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
            with case_(9):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("send_bit", "MOSI")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
            with case_(10):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
            with case_(11):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
            with case_(12):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
            with case_(13):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
            with case_(14):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("wait_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
            with case_(15):
                play("clear", "CS_n", timestamp_stream="clear_play")
                play("tick", "SCLK")
                play("wait_four_cycles","MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
                play("send_bit", "MOSI")
        assign(boo,0)
        with if_(i == (num_rounds-1)):
            align()
            with switch_(boo_final):
                with case_(0):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")  # 5 clock cycles is 1 bit
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                with case_(1):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                with case_(2):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                with case_(3):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                with case_(4):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                with case_(5):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                with case_(6):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                with case_(7):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                with case_(8):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                with case_(9):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                with case_(10):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                with case_(11):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                with case_(12):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                with case_(13):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                with case_(14):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("wait_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                with case_(15):
                    play("clear", "CS_n", timestamp_stream="clear_play")
                    play("tick", "SCLK")
                    play("wait_four_cycles","MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
                    play("send_bit", "MOSI")
        # with if_(i < (num_rounds-1)):
        #     wait(56)

    align()
    wait_for_trigger("TOMESURE")
    play("to_trig", "TOTRIG")
    align()
    # with if_(boo_final > 4):
    play("send_bit","TOMESURE")
    measure("readout", "MEAS_IN", "signal", integration.full("const", out))
    save(out, "out")
    with if_(out<th):
        play("send_bit", "MOSI")



if use_simulator:
    job = qmm.simulate(config, prog, SimulationConfig(int(500*(num_rounds+1))))  # in clock cycles, 4 ns
    samples = job.get_simulated_samples()
    # results = job.result_handles.get('all_bits')
    # print(results.fetch_all())
    # plot it
    print(job.result_handles.clear_play.fetch_all())
    cs_n = samples.con1.digital['3'] + 3
    clk = samples.con1.digital['4'] + 1.5
    mosi = samples.con1.digital['5']
    # to_trig = samples.con1.digital['9'] + 4.5

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set(xlabel="time[ns]")
    ax.plot(cs_n[200:], label="CS_N")
    ax.plot(clk[200:], label="CLK")
    ax.plot(mosi[200:], label="MOSI")
    # ax.plot(to_trig[200:], label="TOTRIG")
    ax.legend()
    ax.set_yticks([])
    ax.yaxis.set_tick_params(labelleft=False)
    ax.tick_params(which='minor', length=4, color='r')
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.grid(which='major', axis='x', linestyle='--')
    ax.grid(which='minor', axis='x', linestyle='--')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


else:
    # qmm = QuantumMachinesManager(host="192.168.116.123", cluster_name='my_cluster_2')  # OPX in research team room
    qm = qmm.open_qm(config)
    job = qm.execute(prog)



## general error probabilities
# ## calculate the error probabilities
# num_rounds=2
# num_bits=4*(num_rounds+1)
# relevant_detectors= list(range(4))
# step = 8
# for i in range(num_rounds-1):
#     relevant_detectors.append(5+i*step)
#     relevant_detectors.append(7 + i * step)
#     relevant_detectors.append(8+i*step)
#     relevant_detectors.append(10 + i * step)
# relevant_detectors= relevant_detectors+list(range(8*num_rounds-4,8*num_rounds))
#
# circuit = stim.Circuit.generated(
#     "surface_code:rotated_memory_z",
#     rounds=num_rounds,
#     distance=3,
#     after_clifford_depolarization= 0.02,
#     before_round_data_depolarization=0,
#     before_measure_flip_probability=0.03)
# model=circuit.detector_error_model()
# rel_probs=np.zeros(len(relevant_detectors))
# probs=np.zeros(model.num_detectors)
# shift=0
# for j in range(len(model)):
#     if type(model[j])==stim._stim_sse2.DemInstruction and model[j].type=='error':
#         for ind in range(len(model[j].targets_copy())):
#             for i in range(model.num_detectors-shift):
#                 if model[j].targets_copy()[ind] == stim.target_relative_detector_id(i):
#                     probs[i+shift] = probs[i+shift]*(1-model[j].args_copy()[0])+(1-probs[i+shift])*model[j].args_copy()[0]
#     if type(model[j])==stim._stim_sse2.DemRepeatBlock: #relevant for large codes
#         for _ in range(model[j].repeat_count):
#             for jj in range(len(model[j].body_copy())):
#                 if model[j].body_copy()[jj].type == 'error':
#                     for ind in range(len(model[j].body_copy()[jj].targets_copy())):
#                         for i in range(model.num_detectors-shift):
#                             if model[j].body_copy()[jj].targets_copy()[ind] == stim.target_relative_detector_id(i):
#                                 probs[i+shift] = probs[i+shift]*(1-model[j].body_copy()[jj].args_copy()[0])+(1-probs[i+shift])*model[j].body_copy()[jj].args_copy()[0]
#                 elif model[j].body_copy()[jj].type == 'shift_detectors':
#                     shift+=model[j].body_copy()[jj].targets_copy()[0]
#
#
# for i,ind in enumerate(relevant_detectors):
#     rel_probs[i]=probs[ind]
#
# print(rel_probs) # we can reduce the calculation significantly since we have symmetry in the problem and that detector probabilities don't change apart of first 4 and last 4 - that is, we need only 12 values, that you can get out of a simulation of 3....
#
