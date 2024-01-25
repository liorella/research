# %%
import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

################################################################################
# simulation parameters - MODIFY THESE AS NEEDED

args = dict(
    theta1=         0.0,    # drive field angle of qubit1 (TBI)
    theta2=         0.0,    # drive field angle of qubit2 (TBI)
    t_start1=       0.0,    # gate start time of qubit 1
    t_start2=       0.0,    # gate start time of qubit 2
    phi1=           0.0,    # carrier phase of qubit 1
    phi2=           0.0,    # carrier phase of qubit 2
    duration=       None,   # gate duration. If set to None, will be set to a full rotation when pi_factor is 1
    pi_factor1=     0.5,    # pi pulse factor of qubit 1
    pi_factor2=     0.5,    # pi pulse factor of qubit 2
    detune_q1=      1,    # drive detuning of qubit 1
    detune_q2=      0.0,    # drive detuning of qubit 2
)

t_final = 6
animation_refresh_rate = 50  # ms
################################################################################

# qubit parameters - advanced modifications
omega1 = 2*np.pi*5  # GHz
omega2 = 2*np.pi*5  # GHz
Delta = omega2 - omega1
omega_av = (omega1 + omega2) / 2
eta = 2*np.pi*0.3  # GHz

# drive parameters
Omega = 2*np.pi*0.2  # GHz
tlist = np.linspace(0, 100, 1000)  # ns

# %%

# operator definitions

I = qp.qeye(2)
sigmax = qp.sigmax()
sigmay = qp.sigmay()
sigmaz = qp.sigmaz()

I2 = qp.tensor([I, I])
sx1 = qp.tensor([sigmax, I])
sy1 = qp.tensor([sigmay, I])
sz1 = qp.tensor([sigmaz, I])
sx2 = qp.tensor([I, sigmax])
sy2 = qp.tensor([I, sigmay])
sz2 = qp.tensor([I, sigmaz])

# static part of interaction Hamiltonian
H01 = omega1 * (sz1 - I2) / 2 + omega2 * (sz2 - I2) / 2

# drive Hamiltonian
Hd1 = Omega * sx1 / 2
Hd2 = Omega * sx2 / 2


# %%

if args['duration'] is None:
    args['duration'] =  2*np.pi / Omega

psi0 = qp.tensor([qp.basis(2, 0), qp.basis(2, 0)])


def flat_pulse(t, t_start, duration):
    if t_start < t and t < t_start + duration:
        return 1
    else:
        return 0


def q1_sigmax_drive(t, args):
    t_start = args['t_start1']
    duration = args['duration']
    pi_factor1 = args['pi_factor1']
    detq1 = args['detune_q1']
    return pi_factor1 * flat_pulse(t, t_start, duration) * np.cos((omega1 + detq1) * t + args['phi1'])


def q2_sigmax_drive(t, args):
    t_start = args['t_start2']
    duration = args['duration']
    pi_factor2 = args['pi_factor2']
    detq2 = args['detune_q2']
    return pi_factor2 * flat_pulse(t, t_start, duration) * np.cos((omega2+detq2) * t + args['phi2'])


H = [H01,
     [Hd1, q1_sigmax_drive],
     [Hd2, q2_sigmax_drive],
     ]

# %%


# %%
# integrate
tvec = np.arange(0, t_final, 0.01)
sol = qp.sesolve(H, psi0, tvec,
                 args=args,
                 options=qp.Options(max_step=1))

e_ops = qp.expect([sx1, sx2, sy1, sy2, sz1, sz2], sol.states)

slist = [[0, 0], [0, 1], [1, 0], [1, 1]]

# plot phases of amplitudes of final states

final_state = np.squeeze(sol.states[-1])
for i, state in enumerate(final_state):
    plt.plot([0, np.real(state)], [0, np.imag(state)],
             'o-', label=str(slist[i]))
plt.title('final state phases')
plt.axis('equal')
plt.legend()

# plt.show()
# plot time trace of drive and prob. amplitudes
fig, ax = plt.subplots(3, 1, sharex='all', figsize=(14, 12))

axn = ax[0]
axn.plot(sol.times, [q1_sigmax_drive(t, args)
         for t in sol.times], label='q1 xy drive')
axn.plot(sol.times, [q2_sigmax_drive(t, args)
         for t in sol.times], label='q2 xy drive')
axn.legend()
axn.set_title('xy drive pulses')
axn.grid('all')

axn = ax[1]
slist = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in range(4):
    axn.plot(sol.times, [np.squeeze(np.abs(s[i])**2) for s in sol.states],
             label=str(slist[i]),
             marker='o',
             markevery=slice(20*i, -1, 80),
             )
axn.legend()
axn.set_ylabel('abs sq.')
axn.grid('all')
axn.set_title('prob. amplitude abs squared')

# phases
axn = ax[2]
# intrinsic phases
for i in range(4):
    axn.plot(sol.times, [np.squeeze(np.angle(s[i]))/np.pi for s in sol.states],
             label=str(slist[i]),
             marker='o',
             markevery=slice(20*i, -1, 80),
             )

# axn.set_yticks(np.arange(-1, 1.25, 0.25))
axn.grid('all')
axn.legend()
axn.set_ylabel('angle/pi')
axn.set_title('prob. amplitude phases')

ax[-1].set_xlabel('time [nsec]')
# ax[-1].set_xlim(0, 1)
# plt.show()
# %%
# %%

frame = 0
fig, ax = plt.subplots(1, 2)  # fig_kw={'num': 1, 'clear': True})

axn = ax[0]
sxy1_plt = axn.plot([0, e_ops[0][frame]], [
                    0, e_ops[2][frame]], 'o-', label='X+iY')
q1_drive_plt = axn.plot([0, q1_sigmax_drive(sol.times[frame], args)], [
                        0, 0], 'o-', label='drive')
axn.add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
axn.set(title='qubit 1', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect=1)

axn = ax[1]
sxy2_plt = axn.plot([0, e_ops[1][frame]], [
                    0, e_ops[3][frame]], 'o-', label='X+iY')
q2_drive_plt = axn.plot([0, q2_sigmax_drive(sol.times[frame], args)], [
                        0, 0], 'o-', label='drive')
axn.add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
axn.set(title='qubit 2', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect=1)


def update(frame):
    sxy1_plt[0].set_data([0, e_ops[0][frame]], [0, e_ops[2][frame]])
    sxy2_plt[0].set_data([0, e_ops[1][frame]], [0, e_ops[3][frame]])
    q1_drive_plt[0].set_data(
        [0, q1_sigmax_drive(sol.times[frame], args)], [0, 0])
    q2_drive_plt[0].set_data(
        [0, q2_sigmax_drive(sol.times[frame], args)], [0, 0])
    return sxy1_plt, sxy2_plt


anim = animation.FuncAnimation(
    fig, update, frames=range(len(e_ops[0])), interval=animation_refresh_rate)
plt.show()
