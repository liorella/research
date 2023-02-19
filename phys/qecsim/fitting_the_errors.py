import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b):
    return a * x + b


##
loaded_data = np.load('NonFT_Shor_repetition_stim.npz')
exp_results = loaded_data.f.exp_results
distance_vec = loaded_data.f.distance_vec
p_vec = loaded_data.f.p_vec
A=np.zeros(len(distance_vec))
B=np.zeros(len(distance_vec))
log_p_vec = np.log10(p_vec)
exp_results+=10**-9*np.ones((len(p_vec), len(distance_vec)))
log_exp_results = np.log10(exp_results)

for k, distance in (enumerate(distance_vec)):
    if k==0:
        popt, pcov = curve_fit(func, log_p_vec[4:9], log_exp_results[4:9, k])
    elif k==1:
        popt, pcov = curve_fit(func, log_p_vec[4:9],log_exp_results[4:9,k])
    elif k == 2:
        popt, pcov = curve_fit(func, log_p_vec[5:9], log_exp_results[5:9, k])
    elif k==3:
        popt, pcov = curve_fit(func, log_p_vec[5:9],log_exp_results[5:9,k])
    A[k]=popt[0]
    B[k]=popt[1]


np.savez('NonFT_Shor_repetition_stim_fitting.npz', A=A, B=B)

##
fig, ax = plt.subplots()
for i, distance in enumerate(distance_vec):
    ax.plot(p_vec, exp_results[:,i], 'o', label=f'distance={distance}')
    ax.plot(10**log_p_vec[4:9], 10**func(log_p_vec[4:9], A[i],B[i]),label=f'distance={distance}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.xscale('log')
plt.yscale('log')


### plotting
p_vec2 = np.logspace(-4.5, -0.5, num=30)
log_p_vec2=np.log10(p_vec2)

NonFT_Shor = np.load('NonFT_Shor_repetition_stim_fitting.npz')
regular = np.load('regular_repetition_stim_fitting.npz')
FT_Reg_3nc = np.load('FT_Reg_3nc_fitting.npz')
FT_Reg_2nc = np.load('FT_Reg_2nc_fitting.npz')
FT_adp_2nc = np.load('FT_adp_2nc_fitting.npz')
FT_adp_3nc = np.load('FT_adp_3nc_fitting.npz')


fig, ax = plt.subplots(2,2)
for i, distance in (enumerate(distance_vec)):

    # ax[i % 2,i // 2].plot(10**log_p_vec2, 10**func(log_p_vec2, NonFT_Shor.f.A[i], NonFT_Shor.f.B[i]),label='Shor style MWPM')
    ax[i % 2,i // 2].plot(10**log_p_vec2, 10**func(log_p_vec2, regular.f.A[i], regular.f.B[i]),label='regular MWPM')
    # ax.plot(10**log_p_vec2, 10**func(log_p_vec2, FT_Reg_3nc.f.A[i], FT_Reg_3nc.f.B[i]),label='Shor style FT 3 anc')
    ax[i % 2,i // 2].plot(10**log_p_vec2, 10**func(log_p_vec2, FT_Reg_2nc.f.A[i], FT_Reg_2nc.f.B[i]),label='Shor style FT 2 anc')
    # ax.plot(10**log_p_vec2, 10**func(log_p_vec2, FT_adp_3nc.f.A[i], FT_adp_3nc.f.B[i]),label='Shor style adaptive 3 anc')
    ax[i % 2,i // 2].plot(10**log_p_vec2, 10**func(log_p_vec2, FT_adp_2nc.f.A[i], FT_adp_2nc.f.B[i]),label='Shor style adaptive 2 anc')
    ax[i % 2,i // 2].plot(10**log_p_vec2, 10**log_p_vec2, label=f'single qubit',linestyle='--', linewidth=1)
    ax[i % 2,i // 2].set_yscale('log')
    ax[i % 2,i // 2].set_xscale('log')
    ax[i % 2,i // 2].set_title(f'distance={distance_vec[i]}')
    ax[i % 2,i // 2].set_ylim((1E-10, 1))
    # plt.legend()
    ax[i % 2,i // 2].grid(linestyle='--', linewidth=0.2)

for ax1 in ax.flat:
    ax1.set(xlabel='physical error', ylabel='logical error')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax1 in ax.flat:
    ax1.label_outer()
plt.show()

##

fig, ax = plt.subplots()
for i, distance in (enumerate(distance_vec)):

    ax.plot(10**log_p_vec2, 10**func(log_p_vec2, FT_adp_2nc.f.A[i], FT_adp_2nc.f.B[i]),label='Shor style adaptive 2 anc')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim((1E-10, 1))
    # plt.legend()
    ax.grid(linestyle='--', linewidth=0.2)

plt.show()

fig, ax = plt.subplots()
for i, distance in (enumerate(distance_vec)):

    ax.plot(10**log_p_vec2, 10**func(log_p_vec2, regular.f.A[i], regular.f.B[i]),label='regular MWPM')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim((1E-10, 1))
    # plt.legend()
    ax.grid(linestyle='--', linewidth=0.2)

plt.show()
##
