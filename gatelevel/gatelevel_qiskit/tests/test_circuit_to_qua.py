import random
from pprint import pprint

import pytest
import qiskit

from gatelevel_qiskit.circuit_to_qua import CircuitQuaTransformer
from gatelevel_qiskit.lib import get_min_time
from gatelevel_qiskit.simple_backend import simple_backend
from gatelevel_qiskit.examples.rb_config import config_base
from qm.qua import *


def test_simple_circuit():
    # Example circuit
    circ = qiskit.QuantumCircuit(2)
    circ.x(0)
    circ.x(0)
    circ.x(0)
    circ.h(1)
    circ.s(0)
    circ.x(1)
    circ.cx(0, 1)

    circ.draw()

    circ_qua = CircuitQuaTransformer(pulse_backend=simple_backend,
                                     config_base=config_base,
                                     circuit=circ)

    wfs_circ = circ_qua.to_waveforms(232.0)
    conf = circ_qua._to_config()
    ret_dict = {}
    exec(circ_qua.to_qua(), globals(), ret_dict)
    print(circ_qua.to_qua())

    from qm.QuantumMachinesManager import QuantumMachinesManager
    from qm import SimulationConfig

    qmm = QuantumMachinesManager()
    qm = qmm.open_qm(conf)

    job = qm.simulate(ret_dict['prog'], SimulationConfig(500, include_analog_waveforms=True))
    print(job.id())
    wfs_sim = job.simulated_analog_waveforms()['controllers']['con1']['ports']

    from deepdiff import DeepDiff
    diff = DeepDiff(wfs_sim, wfs_circ, significant_digits=5)
    print("old value - QUA simulator\nnew value - circuit output")
    pprint(diff)
    assert len(diff) == 0


@pytest.mark.parametrize("seed", range(10))
def test_random_circuit(seed):
    # random circuit
    random.seed(seed)
    circ_len = 10
    from qiskit.circuit.library import XGate, HGate, CXGate, YGate, SGate
    gateset = [CXGate(), XGate(), HGate(), YGate(), SGate()]

    circ = qiskit.QuantumCircuit(2)
    for _ in range(circ_len):
        gate = random.choice(gateset)
        if isinstance(gate, CXGate):
            qbs = [0, 1]
        else:
            qbs = random.choice([[0], [1]])
        circ.append(gate, qbs)

    print("")
    print(circ.draw())
    circ_qua = CircuitQuaTransformer(pulse_backend=simple_backend,
                                     config_base=config_base,
                                     circuit=circ)

    ret_dict = {}
    exec(circ_qua.to_qua(), globals(), ret_dict)
    print(circ_qua.to_qua())

    from qm.QuantumMachinesManager import QuantumMachinesManager
    from qm import SimulationConfig

    qmm = QuantumMachinesManager()
    qm = qmm.open_qm(circ_qua.config)

    job = qm.simulate(ret_dict['prog'], SimulationConfig(500, include_analog_waveforms=True))
    print(job.id())
    wfs_sim = job.simulated_analog_waveforms()['controllers']['con1']['ports']

    wfs_circ = circ_qua.to_waveforms(get_min_time(wfs_sim))
    from deepdiff import DeepDiff
    diff = DeepDiff(wfs_sim, wfs_circ, significant_digits=5)
    print("old value - QUA simulator\nnew value - circuit output")
    pprint(diff)
    assert len(diff) == 0


@pytest.mark.parametrize("seed", range(10))
def test_rb_circuit(seed):
    from gatelevel_qiskit.waveform_comparator import WaveformComparator

    # Import the RB Functions
    import qiskit.ignis.verification.randomized_benchmarking as rb
    # Import Qiskit classes
    import qiskit
    from pprint import pprint

    from gatelevel_qiskit.circuit_to_qua import CircuitQuaTransformer
    from gatelevel_qiskit.lib import get_min_time
    from gatelevel_qiskit.simple_backend import simple_backend
    from gatelevel_qiskit.examples.rb_config import config_base

    # generate RB 1QB
    c1 = qiskit.circuit.quantumcircuit.QuantumCircuit(1)
    c1.x(0)
    rb_circs1, xdata = rb.randomized_benchmarking_seq(length_vector=[1, 2, 3, 4, 5],
                                                      nseeds=1,
                                                      rb_pattern=[[0]],
                                                      rand_seed=seed)
    circ = rb_circs1[0][4]
    print("\n")
    print(circ.draw())

    circ_qua = CircuitQuaTransformer(pulse_backend=simple_backend,
                                     config_base=config_base,
                                     circuit=circ)

    from qm.QuantumMachinesManager import QuantumMachinesManager
    from qm import SimulationConfig

    qmm = QuantumMachinesManager()
    qm = qmm.open_qm(circ_qua.config)

    ret_dict = {}
    exec(circ_qua.to_qua(), globals(), ret_dict)
    job = qm.simulate(ret_dict['prog'], SimulationConfig(500, include_analog_waveforms=True))
    print(job.id())
    wfs_sim = job.simulated_analog_waveforms()['controllers']['con1']['ports']

    wfs_circ = circ_qua.to_waveforms(get_min_time(wfs_sim))

    comp = WaveformComparator(wfs_sim, wfs_circ)
    pprint(comp)
    assert len(comp.diff) == 0
