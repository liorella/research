import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import numpy as np
import qutip
from pymatching import Matching
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from vlogging import VisualRecord

from qecsim.lib import quantumsim_dm_to_qutip_dm, SimDataHandler
from qecsim.surface_code_generator import SurfaceCodeGenerator
from qecsim.qec_generator import CircuitParams


sdh = SimDataHandler()
plot = False
sdh.log.setLevel(logging.INFO)
# distance = 4
encoded_state = '1'
cparams = CircuitParams(t1=15e3,
                        t2=19e3,
                        single_qubit_gate_duration=20,
                        two_qubit_gate_duration=100,
                        single_qubit_depolarization_rate=1.1e-3,
                        two_qubit_depolarization_rate=6.6e-3,
                        meas_duration=600,
                        reset_duration=0,
                        reset_latency=40)
generator = SurfaceCodeGenerator(3, cparams)
state = quantumsim.sparsedm.SparseDM(generator.register_names)
results_record = []

print('starting')
generator.generate_stabilizer_round().apply_to(state)
results_record.append(state.classical[cb] for cb in [f'm{q}' for q in generator.ancillas])
generator.generate_stabilizer_round(final_round=True).apply_to(state)
results_record.append(state.classical[cb] for cb in generator.cbit_names)

print(results_record)




plt.show()

