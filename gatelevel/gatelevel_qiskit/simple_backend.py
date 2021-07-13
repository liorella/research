import numpy as np
from qiskit.circuit import Barrier
from qiskit.pulse import Schedule, Play, Gaussian, DriveChannel, ShiftPhase, Waveform, MeasureChannel, ConstantPulse, \
    Constant, Acquire, AcquireChannel, MemorySlot
from qiskit.scheduler import measure
from qiskit.test.mock import FakeOpenPulse2Q, FakeAlmaden

from gatelevel_qiskit.pulse_backend import PulseBackend

# single qubit gates
X_q = [Schedule(Play(Gaussian(duration=20, amp=0.2, sigma=2).get_waveform(),
                     name='wf_X',
                     channel=DriveChannel(nq)))
       for nq in range(2)]

Y_q = [Schedule(Play(Gaussian(duration=20, amp=0.2 * 1j, sigma=2).get_waveform(),
                     name='wf_Y',
                     channel=DriveChannel(nq)))
       for nq in range(2)]  # todo: plus or minus?

H_q = []
for nq in range(2):
    H_q.append(Schedule(ShiftPhase(np.pi / 2, channel=DriveChannel(nq))))  # todo: plus or minus?
    H_q[nq] += Play(Gaussian(duration=20, amp=0.1, sigma=2).get_waveform(), name='wf_X90', channel=DriveChannel(nq))
    H_q[nq] += ShiftPhase(np.pi / 2, channel=DriveChannel(nq))

S_q = [Schedule(ShiftPhase(np.pi / 2, channel=DriveChannel(nq))) for nq in range(2)]

Z_q = [Schedule(ShiftPhase(np.pi, channel=DriveChannel(nq))) for nq in range(2)]
Sdg_q = [Schedule(ShiftPhase(-np.pi / 2, channel=DriveChannel(nq))) for nq in range(2)]

Barrier_q = Barrier(num_qubits=2)

# adapt CX from Almaden
almaden_be = FakeAlmaden()
almaden_cx_01 = almaden_be.defaults().instruction_schedule_map.get('cx', [0, 1])
almaden_dt = almaden_be.configuration().dt * 1e9

cx_01 = Schedule()
al_insts = iter(almaden_cx_01.instructions)
inst = next(al_insts)
cx_01 += inst[1]
inst = next(al_insts)
cx_01 += inst[1]
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample and scale
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel).shift(160 // 8)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel).shift(512 // 8)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel).shift(160 // 8)
inst = next(al_insts)
samples_ds = inst[1].pulse.samples[::8] / 2  # downsample
cx_01 += Play(Waveform(samples_ds, name=inst[1].pulse.name), channel=inst[1].channel).shift(160 // 8)

backend = FakeOpenPulse2Q()

defaults = backend.defaults()
cals = defaults.instruction_schedule_map

simple_backend = PulseBackend(backend, cals)

for nq in range(2):
    simple_backend.add('x', nq, X_q[nq])
    simple_backend.add('y', nq, Y_q[nq])
    simple_backend.add('h', nq, H_q[nq])
    simple_backend.add('s', nq, S_q[nq])
    simple_backend.add('sdg', nq, Sdg_q[nq])
    simple_backend.add('z', nq, Z_q[nq])
simple_backend.add('cx', [0, 1], cx_01)
