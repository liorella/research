import numpy as np
import qiskit
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import ProcessTomography, StateTomography

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeParis

# Noisy simulator backend
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis

backend = AerSimulator.from_backend(FakeParis())

# Run experiments

# GHZ State preparation circuit
nq = 1
qc_ghz = qiskit.QuantumCircuit(nq)
qc_ghz.h(0)
qc_ghz.s(0)
# for i in range(1, nq):
#     qc_ghz.cx(0, i)

# QST Experiment
qstexp1 = StateTomography(qc_ghz)
qstdata1 = qstexp1.run(backend, seed_simulation=100).block_for_results()

from qiskit_experiments.library.tomography.qst_analysis import StateTomographyAnalysis

sta = StateTomographyAnalysis()

from qiskit_experiments.library.tomography.fitters import linear_inversion

linv = linear_inversion(np.array([[100, 200], [50, 250], [30, 270]]),
                        np.array([300, 300, 300]),
                        np.array([[0], [1], [2]]),
                        np.array([]),
                        measurement_basis=PauliMeasurementBasis(),
                        measurement_qubits=(0, ))
