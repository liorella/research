import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from qec_generator import CircuitParams
from simulate_qec_rounds_stim import experiment_run
from stim_lib.scheduled_shor_nonFT_circuit import generate_shor_nonFT_scheduled
from stim_lib.scheduled_circuit import generate_scheduled
from scipy.optimize import curve_fit
##
 matplotlib.rcParams['font.size'] = 14
def log_err_fit(rounds, e_logical):
    return 0.5*(1 - (1-2*e_logical)**rounds)

cparams = CircuitParams(t1=15e3,
                t2=8e3,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=20,
                single_qubit_depolarization_rate=1e-3,
                two_qubit_depolarization_rate=5e-3,
                meas_duration=550,
                reset_duration=0,
                reset_latency=40,
                meas_induced_dephasing_enhancement=3)


distance_vec = [3, 5, 7]
rounds_vec = np.arange(2, 10, 2)
shots=2000
distance=distance_vec[1]
rounds=rounds_vec[1]
##
def get_error_rate_log_err_vec(reset_strategy):
#     circ, cont = generate_scheduled(
#     code_task='surface_code:rotated_memory_z',  # looks ok
#     distance=3,
#     rounds=1,
#     params=cparams
#     )
#     print(circ)
    log_err_vec = []
    error_rate = []

    print(cparams.reset_duration)
    for distance in distance_vec:
        error_rate.append([])
        print(f'starting distance = {distance}')
        for rounds in tqdm(rounds_vec):
            circ, cont, _ = generate_shor_nonFT_scheduled(code_task='rep_Shor_nonFT',
                distance=distance,
                rounds=rounds,
                params=cparams
            )
            error_rate[-1].append(1 - experiment_run(circ, cont, shots=shots, reset_strategy=reset_strategy))
        log_err, pcov = curve_fit(log_err_fit, rounds_vec, error_rate[-1], p0=0)
        log_err_vec.append(log_err)

    log_err_vec = np.array(log_err_vec).squeeze()
    error_rate = np.array(error_rate)
    return error_rate, log_err_vec


def plot_lambda(error_rate, log_err_vec, title, filename):
    plt.figure(figsize=(6, 5))
    for i, distance in enumerate(distance_vec):
        plt.plot(rounds_vec, error_rate[i], 'o', label=f'distance={distance}')
        plt.plot(rounds_vec, log_err_fit(rounds_vec, log_err_vec[i]), '--')

    plt.legend()
    plt.xlabel('number of rounds')
    plt.ylabel('logical error rate')
    plt.title(title)
    plt.ylim((0, 0.2))
    print(f'1/Lambda = {log_err_vec[1] / log_err_vec[0]}')
    plt.savefig(filename)
## Extracting 1/Lambda with AR
cparams.reset_duration=300
error_rate_ar, log_err_vec_ar = get_error_rate_log_err_vec('AR')

plot_lambda(
    error_rate_ar,
    log_err_vec_ar,
    f'AR, $1/\Lambda$={log_err_vec_ar[1] / log_err_vec_ar[0]:.2f}, reset duration={cparams.reset_duration}',
    'AR_RD=300_GE_hi.svg'
)

##Active parity tracking benefit
cparams.reset_duration=0
error_rate_apt, log_err_vec_apt = get_error_rate_log_err_vec('APT')
plot_lambda(
    error_rate_apt,
    log_err_vec_apt,
    f'APT, $1/\Lambda$={log_err_vec_apt[1] / log_err_vec_apt[0]:.2}, reset duration={cparams.reset_duration}',
    'APT_RD=0_GE_hi.svg'
)

## What happens when there are no gate errors at all?

cparams.reset_duration=0
cparams.single_qubit_depolarization_rate = 1e-4
cparams.two_qubit_depolarization_rate = 1e-4
error_rate_apt_low, log_err_vec_apt_low = get_error_rate_log_err_vec('APT')

plot_lambda(
    error_rate_apt_low,
    log_err_vec_apt_low,
    f'APT, GE=1e-4, $1/\Lambda$={log_err_vec_apt_low[1] / log_err_vec_apt_low[0]:.2}, reset duration={cparams.reset_duration}',
    'APT_RD=0_GE_1e-4.svg'
)

##
cparams.reset_duration=300
cparams.single_qubit_depolarization_rate = 1e-4
cparams.two_qubit_depolarization_rate = 1e-4
error_rate_ar_low, log_err_vec_ar_low = get_error_rate_log_err_vec('AR')
plot_lambda(
    error_rate_ar_low,
    log_err_vec_ar_low,
    f'AR, GE=1e-4, $1/\Lambda$={log_err_vec_ar_low[1] / log_err_vec_ar_low[0]:.2}, reset duration={cparams.reset_duration}',
    'AR_RD=300_GE_1e-4.svg'
)
