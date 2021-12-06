from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from gatelevel_qiskit.examples.qv_config import config_base, num_qubits

shots = 100
K = 10
I_thresh = [0 for _ in range(num_qubits)]


def assign_random_parameters(params):
    with for_(i, 0, i < num_qubits * 3, i + 1):
        assign(params[i], r.rand_fixed())


def initialize_qubits():
    I = [declare(fixed) for _ in range(num_qubits)]
    for i in range(num_qubits):
        measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))
        with while_(I[i] > I_thresh[i]):
            # play('pi_pulse', f'q{i}')  # todo: uncomment when done
            measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))


def run_qv_circuit(params):
    pass


def measure_state(state):
    I = [declare(fixed) for _ in range(num_qubits)]
    for i in range(num_qubits):
        measure('readout_pulse', f'm{i}', None, dual_demod.full('iw1', 'out1', 'iw2', 'out2', I[i]))
        assign(state[i], I[i] > I_thresh[i])

    # randomize measurement result (todo: remove when done)
    i = declare(int)
    with for_(i, 0, i < num_qubits, i + 1):
        assign(qubits_state[i], Cast.to_bool(r.rand_int(2)))
        save(qubits_state[i], 'qs')


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


with program() as prog:
    parameters = declare(fixed, size=num_qubits * 3)
    qubits_state = declare(bool, size=num_qubits)
    k = declare(int)  # iteration number
    i = declare(int)
    n_shots = declare(int)
    seed = declare(int, value=0)
    r = lib.Random()
    r.set_seed(seed)
    assign_random_parameters(parameters)

    with for_(k, 0, k < K, k + 1):
        with for_(n_shots, 0, n_shots < shots, n_shots + 1):
            initialize_qubits()
            run_qv_circuit(parameters)
            measure_state(qubits_state)
            bool_to_int(seed, qubits_state, num_qubits)
            save(seed, 'seed')
            r.set_seed(seed)
            assign_random_parameters(parameters)

qmm = QuantumMachinesManager('192.168.1.119')
job = qmm.simulate(config_base, prog, SimulationConfig(2200))
job.result_handles.wait_for_all_values()
qs_res = job.result_handles.get('qs').fetch_all()['value']
seed_res = job.result_handles.get('seed').fetch_all()['value']

qs_res = qs_res.reshape((-1, num_qubits))
print(qs_res)
print(seed_res)
