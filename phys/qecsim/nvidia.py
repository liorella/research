import numpy as np
from matplotlib import pyplot as plt
## first stage

M = 4
d=np.linspace(3,9,4)

N=M*(2*d**2+4*d)

data_out=2.5*M*np.ones(4)
data_in=N/2
latency=d

##
fig1, (ax2,ax1,ax3) = plt.subplots(1,3)
ax1.plot(N, data_out, 'o', label='increasing d with M=4')
ax2.plot(N, data_in, 'o')
ax3.plot(N, latency, 'o')

## second stage
d=9

M=np.linspace(4,15,12)

N=M*(2*d**2+4*d)

data_out=2.5*M
data_in=N/2
latency=d*np.ones(12)

ax1.plot(N, data_out, 'o', label='increasing M with d=9')
ax2.plot(N, data_in, 'o')
ax3.plot(N, latency, 'o')



## third stage

d=np.linspace(11,15,3)
M=15
N=M*(2*d**2+4*d)

data_out=2.5*M*np.ones(3)

data_in=N/2
latency=d

ax1.plot(N, data_out, 'o', label='increasing d with M=15')
ax2.plot(N, data_in, 'o')
ax3.plot(N, latency, 'o')

## 4rth stage
d=15

M=np.linspace(16,50,20)

N=M*(2*d**2+4*d)

data_out=2.5*M
data_in=N/2
latency=d*np.ones(20)

ax1.plot(N, data_out, 'o', label='increasing M with d=15')
ax2.plot(N, data_in, 'o')
ax3.plot(N, latency, 'o')
##
ax1.set_xlabel('number of physical qubits')
ax1.set_ylabel('maximal decoder output [bits/microsecond]')

ax2.set_xlabel('number of physical qubits')
ax2.set_ylabel('maximal  input to the decoder [bits/microsecond]')

ax3.set_xlabel('number of physical qubits')
ax3.set_ylabel('decoder latency [microsecond]')

ax1.legend()
ax1.set_xscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')

ax3.set_xscale('log')

plt.rc('font', size=14)          # controls default text sizes
