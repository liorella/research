import time

import numpy as np
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate.credentials import create_credentials

from gatelevel_qiskit.clops_maker import QVMaker
from gatelevel_qiskit.examples.opx_connectivity import create_fully_connected
from gatelevel_qiskit.examples.qv_config import num_qubits

shots = 100
K = 10
M = 10
I_thresh = [0 for _ in range(num_qubits)]
num_templates_per_prog = 1


def assign_random_parameters(params):
    i = declare(int)
    with for_(i, 0, i < num_qubits * 3, i + 1):
        assign(params[i], r.rand_fixed())
        save(params[i], 'xxx')


def initialize_qubits():
    I = [declare(fixed) for _ in range(num_qubits)]
    for i in range(num_qubits):
        measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))
        align(f'm{i}', f'd{i}')
        with if_(I[i] > -2.0):
            play('wf_sx', f'd{i}')
            # align(f'm{i}', f'd{i}')
            # measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))


def measure_state(state):
    I = [declare(fixed) for _ in range(num_qubits)]
    for i in range(num_qubits):
        measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))
        assign(state[i], I[i] > I_thresh[i])
        save(state[i], 'qs')

    # randomize measurement result (todo: remove when done)
    # i = declare(int)
    # with for_(i, 0, i < num_qubits, i + 1):
    #     assign(qubits_state[i], Cast.to_bool(r.rand_int(2)))


def bool_to_int(int_result, bool_vector, binary_len):
    """
    Convert a boolean array to an integer whose binary representation is equal to the array

    Assumes that the boolean has length
    :param int_result: result (qua integer variable, modified in place)
    :param bool_vector: the input array (qua boolean array)
    :param binary_len: the length of the binary array (qua integer)
    :return: None
    """
    assign(int_result, 0)
    i_bi = declare(int)
    with for_(i_bi, 0, i_bi < binary_len, i_bi + 1):
        assign(int_result, int_result + (Cast.unsafe_cast_int(bool_vector[i_bi]) << i_bi))


parameters = []

print('creating programs...')
progs = []
for p_i in range(M // num_templates_per_prog):
    print(f'creating program for circuits {p_i * num_templates_per_prog} to {(p_i + 1) * num_templates_per_prog - 1}')
    with program() as prog:
        parameters = declare(fixed, size=num_qubits * 3)
        qubits_state = declare(bool, size=num_qubits)
        k = declare(int)  # iteration number
        n_shots = declare(int)
        seed = declare(int, value=0)
        r = lib.Random()
        r.set_seed(seed)
        assign_random_parameters(parameters)

        finished = declare(bool, value=False)
        save(finished, 'finished')
        for _ in range(num_templates_per_prog):
            align()
            with for_(k, 0, k < K, k + 1):
                with for_(n_shots, 0, n_shots < shots, n_shots + 1):
                    initialize_qubits()
                    qvm = QVMaker()
                    qv_circuit = qvm.make_qv_macro()
                    qv_circuit(parameters)
                    measure_state(qubits_state)
                    bool_to_int(seed, qubits_state, num_qubits)
                r.set_seed(seed)
                assign_random_parameters(parameters)
        assign(finished, True)
        save(finished, 'finished')
    progs.append(prog)

qmm = QuantumMachinesManager('192.168.1.119')

# qmm = QuantumMachinesManager(host='oded-36a11cb2.dev.quantum-machines.co', port=443, credentials=create_credentials())
simulate = True
if simulate:
    job = qmm.simulate(qvm.config,
                       prog,
                       SimulationConfig(1200,
                                        controller_connections=create_fully_connected(
                                            controllers=['con1', 'con2', 'con3'])),
                       flags=['auto-element-thread'])
    job.result_handles.wait_for_all_values()
else:
    qm = qmm.open_qm(qvm.config)
    print('compiling programs...')
    pids = [qm.compile(progs[i], flags=['auto-element-thread']) for i in range(M // num_templates_per_prog)]
    print('done.')
    tic = time.time()
    for j in range(M // num_templates_per_prog):
        print(f'starting job for circuits {j * num_templates_per_prog} to {(j + 1) * num_templates_per_prog - 1}')
        pjob = qm.queue.add_compiled(pids[j])
        job = pjob.wait_for_execution()
        job.result_handles.wait_for_all_values()
        qs_res = job.result_handles.get('qs').fetch_all()['value']
        finished_res = job.result_handles.get('finished').fetch_all()
    toc = time.time() - tic
    print('total time: ', toc)
    print('CLOPS:', (K * shots * (M // num_templates_per_prog) * num_templates_per_prog * num_qubits) / toc)
    print('circuit time + delay time: ',
          np.diff(finished_res['timestamp']) / 1000 / num_templates_per_prog / n_shots / K,
          'usec')

# qs_res = job.result_handles.get('qs').fetch_all()['value']
# seed_res = job.result_handles.get('seed').fetch_all()['value']

if simulate:
    job.get_simulated_samples().con1.plot()
    job.get_simulated_samples().con2.plot()

# qs_res = qs_res.reshape((-1, num_qubits))
# print(qs_res)
# print(seed_res)
