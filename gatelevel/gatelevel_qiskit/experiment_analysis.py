import uuid

import xarray as xr
from qiskit.providers import Job, Backend, JobStatus
from qiskit.qobj import QobjExperimentHeader
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData


class QuaJob(Job):
    def __init__(self, result: Result):
        job_id = uuid.uuid4().hex  # todo: see if QmJob has an id
        self._job_id = 0
        self._backend = Backend()
        super().__init__()
        self._result = result
        self._status = JobStatus.DONE

    def result(self):
        """Return job result."""
        return self._result

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

    def status(self):
        return self._status


def outcome_da_to_qiskit_result(counts_ar: xr.DataArray, header: QobjExperimentHeader) -> Result:
    """
    convert an xarray dataset into a qiskit job
    :param header: an experiment-specific header
    :param counts_ar: a data array with the dim `outcome` on the first axis and a coord vec of the form '0xXX' which
                    specifies the measured bit vector.
    :return: a result object which contains the measured results as a list of `ExperimentResult` objects
    """
    counts_ar = counts_ar.stack(others=counts_ar.dims[1:])
    counts_all = counts_ar.groupby('others')
    shots_all = counts_all.sum(...)
    results = []
    for shots, counts in zip(shots_all, counts_all):
        results.append(ExperimentResult(shots=int(shots.values),
                                        success=True,
                                        data=ExperimentResultData({c.outcome.item(): int(c.values) for c in counts[1]}),
                                        meas_level=2,
                                        header=header
                                        ))
    return Result('name', 0, 0, 0, True, results=results)