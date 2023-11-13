"""
intro_to_integration.py: Demonstrate usage of the integration in the measure statement
Author: Gal Winer - Quantum Machines
Created: 31/12/2020
Revised by Tomer Feld - Quantum Machines
Revision date: 24/04/2022
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
from configuration_demod import *
import matplotlib.pyplot as plt

# Open communication with the server.
qop_ip = None
qmm = QuantumMachinesManager(host="192.168.116.123", cluster_name='my_cluster_2') # OPX in research team room
##
# Sliced demodulation parameters
num_segments = 25
seg_length = readout_len // (4 * num_segments)


with program() as measureProg:
    ind = declare(int)

    int_stream_I = declare_stream()
    int_stream_Q = declare_stream()
    sliced_demod_res_I = declare(fixed, size=int(num_segments))
    sliced_demod_res_Q = declare(fixed, size=int(num_segments))
    reset_phase("qe1")
    measure("readout", "qe1", None, demod.sliced("cos", sliced_demod_res_I, seg_length),  demod.sliced("sin", sliced_demod_res_Q, seg_length))
    with for_(ind, 0, ind < num_segments, ind + 1):  # save a QUA array
        save(sliced_demod_res_I[ind], int_stream_I)
        save(sliced_demod_res_Q[ind], int_stream_Q)

    with stream_processing():
        int_stream_I.save_all("demod_sliced_I")
        int_stream_Q.save_all("demod_sliced_Q")


job = qmm.simulate(
    config,
    measureProg,
    SimulationConfig(4000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)

res = job.result_handles
# sliced_I = res.demod_sliced_I.fetch_all()["value"]
# print(f"Result of sliced demodulation in {num_segments} segments:{sliced_I}")
##
[f, ax1] = plt.subplots()
ax1.plot(res.demod_sliced_I.fetch_all(), "o-")
ax1.set_title("sliced demod I and Q")
ax1.set_xlabel("slice number")
ax1.plot(res.demod_sliced_Q.fetch_all(), "o-")

