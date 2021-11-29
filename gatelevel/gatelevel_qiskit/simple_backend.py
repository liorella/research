import numpy as np
from qiskit.circuit import Barrier, Parameter
from qiskit.pulse import Schedule, Play, Gaussian, DriveChannel, ShiftPhase, Waveform, MeasureChannel, \
    Constant, Acquire, AcquireChannel, MemorySlot, ControlChannel
from qiskit.scheduler import measure
from qiskit.test.mock import FakeOpenPulse2Q, FakeAlmaden

from gatelevel_qiskit.pulse_backend import PulseBackend

# parameters
num_qubits = 4
qubit_coupling_map = [[0, 1], [1, 2], [2, 3]]

# create backend
backend = FakeOpenPulse2Q()

defaults = backend.defaults()
cals = defaults.instruction_schedule_map

simple_backend = PulseBackend(backend, cals)

# single qubit gates
X_q = [Schedule(Play(Gaussian(duration=20, amp=0.2, sigma=2).get_waveform(),
                     name='wf_X',
                     channel=DriveChannel(nq)))
       for nq in range(num_qubits)]

Y_q = [Schedule(Play(Gaussian(duration=20, amp=0.2 * 1j, sigma=2).get_waveform(),
                     name='wf_Y',
                     channel=DriveChannel(nq)))
       for nq in range(num_qubits)]  # todo: plus or minus?

sx_q = [Schedule(Play(Gaussian(duration=20, amp=0.1, sigma=2).get_waveform(),
                      name='wf_sx',
                      channel=DriveChannel(nq)))
        for nq in range(num_qubits)]

H_q = []
for nq in range(num_qubits):
    H_q.append(Schedule(ShiftPhase(np.pi / 2, channel=DriveChannel(nq))))
    H_q[nq] += Play(Gaussian(duration=20, amp=0.1, sigma=2).get_waveform(), name='wf_X90', channel=DriveChannel(nq))
    H_q[nq] += ShiftPhase(np.pi / 2, channel=DriveChannel(nq))

S_q = [Schedule(ShiftPhase(np.pi / 2, channel=DriveChannel(nq))) for nq in range(num_qubits)]
Z_q = [Schedule(ShiftPhase(np.pi, channel=DriveChannel(nq))) for nq in range(num_qubits)]
Sdg_q = [Schedule(ShiftPhase(-np.pi / 2, channel=DriveChannel(nq))) for nq in range(num_qubits)]
u1_q = [Schedule(ShiftPhase(Parameter('phi'), channel=DriveChannel(nq))) for nq in range(num_qubits)]

Barrier_q = Barrier(num_qubits=num_qubits)

for nq in range(num_qubits):
    simple_backend.add('x', nq, X_q[nq])
    simple_backend.add('y', nq, Y_q[nq])
    simple_backend.add('h', nq, H_q[nq])
    simple_backend.add('s', nq, S_q[nq])
    simple_backend.add('sdg', nq, Sdg_q[nq])
    simple_backend.add('z', nq, Z_q[nq])
    simple_backend.add('sx', nq, sx_q[nq])
    simple_backend.add('u1', nq, u1_q[nq])

# adapt CX from Almaden
almaden_be = FakeAlmaden()
for cpl in qubit_coupling_map:
    # we take the CX between qubits 0 and 1 in Almaden. I am not sure how generic this is... Because for example between
    # 1 and 2 also has additional set phase pulses that I am not sure what is their origin
    almaden_cx_01 = almaden_be.defaults().instruction_schedule_map.get('cx', [0, 1])
    almaden_dt = almaden_be.configuration().dt * 1e9

    cx = Schedule()
    al_insts = iter(almaden_cx_01.instructions)
    # 0
    inst = next(al_insts)  # (0, ShiftPhase(1.5707963267948966, DriveChannel(0)))
    cx += ShiftPhase(np.pi / 2, DriveChannel(cpl[0]))
    # 1
    inst = next(al_insts)  # (0, ShiftPhase(1.5707963267948966, ControlChannel(1)))
    cx += ShiftPhase(np.pi / 2, ControlChannel(cpl[1]))
    # 2
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=DriveChannel(cpl[0]))
    # 3
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=DriveChannel(cpl[1]))
    # 4
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=DriveChannel(cpl[1]))
    # 5
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=ControlChannel(cpl[0])).shift(160 // 8)
    # 6
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=DriveChannel(cpl[0])).shift(512 // 8)
    # 7
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=DriveChannel(cpl[1])).shift(160 // 8)
    # 8
    inst = next(al_insts)
    samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
    cx += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=ControlChannel(cpl[0])).shift(160 // 8)

    simple_backend.add('cx', cpl, cx)
