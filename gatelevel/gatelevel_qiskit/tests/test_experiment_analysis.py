import numpy as np
import xarray as xr
from pytest import fixture
from qiskit import QuantumCircuit
from qiskit.qobj import QobjExperimentHeader
from qiskit.quantum_info import DensityMatrix
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import StateTomography
from qiskit_experiments.library.tomography import StateTomographyAnalysis
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment

from gatelevel_qiskit.experiment_analysis import QuaJob, outcome_da_to_qiskit_result


@fixture
def ds_1q_tomo():
    num_q = 1
    n_avg = 100
    return xr.Dataset({
        'counts': xr.DataArray(np.array([[100, 0, 0], [0, 0, 0]]),
                               dims=['outcome', 'measbase'],
                               coords={'outcome': [hex(c) for c in range(2 ** num_q)], 'measbase': ['Z', 'X', 'Y']})
    },
        attrs={'n_avg': n_avg, 'qubits': [0]})


@fixture
def default_header():
    return QobjExperimentHeader(
        metadata={'clbits': [0], 'm_idx': [0]})  # todo: see how this should vary based on which qubit we are using


def test_create_qiskit_result(ds_1q_tomo, default_header):
    result = outcome_da_to_qiskit_result(ds_1q_tomo.counts,
                                         default_header)
    assert isinstance(result, Result)
    assert len(result.results) == 3
    for i, res in enumerate(result.results):
        assert isinstance(res, ExperimentResult)
        assert res.shots == ds_1q_tomo.attrs['n_avg']
        assert isinstance(res.data, ExperimentResultData)
        assert res.data.counts['0x0'] == ds_1q_tomo.counts.isel(measbase=i).sel(outcome='0x0')
        assert res.data.counts['0x1'] == ds_1q_tomo.counts.isel(measbase=i).sel(outcome='0x1')


def test_create_qiskit_job(ds_1q_tomo, default_header):
    job = QuaJob(outcome_da_to_qiskit_result(ds_1q_tomo.counts, default_header))
    assert isinstance(job.result(), Result)


def test_create_qiskit_experiment_data(ds_1q_tomo, default_header):
    expdat = ExperimentData(TomographyExperiment(QuantumCircuit(1),

                                                 measurement_qubits=[1],
                                                 measurement_basis=PauliMeasurementBasis(),
                                                 qubits=range(5)))
    print(expdat.metadata)
    expdat.add_jobs([QuaJob(outcome_da_to_qiskit_result(ds_1q_tomo.counts, default_header))])


def test_analyze_tomo_1q(ds_1q_tomo, default_header):
    tomo_experiment = StateTomography(QuantumCircuit(1), measurement_qubits=[1],
                                      measurement_basis=PauliMeasurementBasis(), qubits=range(5))
    expdat = ExperimentData(tomo_experiment)
    expdat.add_jobs([QuaJob(outcome_da_to_qiskit_result(ds_1q_tomo.counts, default_header))])
    expdat = StateTomographyAnalysis().run(expdat)
    print(expdat)
    print(expdat.analysis_results())
    assert expdat.analysis_results()[0].value == DensityMatrix([[1. + 0.j, 0. + 0.j],
                                                                [0. + 0.j, 0. + 0.j]],
                                                               dims=(2,))
    pass
    # todo: check that we have the density matrix and the fidelity
