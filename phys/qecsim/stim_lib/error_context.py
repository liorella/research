from itertools import takewhile, dropwhile

import numpy as np
import pymatching
import stim

from qec_generator import CircuitParams


def _get_records(circ: stim.Circuit) -> list:
    m = []
    for inst in circ:
        if isinstance(inst, stim.CircuitRepeatBlock):
            for _ in range(inst.repeat_count):
                m.extend(_get_records(inst.body_copy()))
        elif inst.name in ('M', 'MR'):
            m.extend(t.value for t in inst.targets_copy())
    return m


class StimErrorContext:
    """
    Generates a context for decoding and running error correction sequences on generated stim_lib QEC circuits
    """

    def __init__(self, circuit: stim.Circuit, code_task: str, distance: int, rounds: int, params: CircuitParams):
        self._circuit = circuit

        measures = _get_records(self._circuit)
        self._match_indices = []
        self._code_task = code_task
        self._distance = distance
        self._rounds = rounds
        self._params = params
        # this assumes that the reversed structure of the generated program is logical observables and then
        # detectors with parities on the data qubits

        # get matching matrix
        last_detectors = takewhile(lambda x: x.name == 'DETECTOR',
                                   dropwhile(lambda x: x.name != 'DETECTOR',
                                             reversed(self._circuit)))

        for detector in last_detectors:
            self._match_indices.append([measures[k] for k in [t.value for t in detector.targets_copy()]])
        self._ancillas = sorted(r[-1] for r in self._match_indices)
        data_qubits = []
        for r in self._match_indices:
            data_qubits.extend(r[:-1])
        self._data_qubits = sorted(set(data_qubits))

        # get logical operators
        self._logical_observables = []
        logical_observables = takewhile(lambda x: x.name == 'OBSERVABLE_INCLUDE', reversed(self._circuit))
        for obs in logical_observables:
            self._logical_observables.append([measures[k] for k in [t.value for t in obs.targets_copy()]])
        self._build_pymatch_obj()

    @property
    def active_ancillas(self) -> np.ndarray:
        """
        these are the ancillas used for error correction and are needed for decoding
        :return:
        """
        return np.array(self._ancillas)

    @property
    def data_qubits(self) -> np.ndarray:
        return np.array(self._data_qubits)

    @property
    def matching_matrix(self) -> np.ndarray:
        match_matrix = np.zeros((len(self._ancillas), len(self._data_qubits)), dtype=np.uint8)
        for r in self._match_indices:
            match_matrix[self._ancillas.index(r[-1]), [self._data_qubits.index(i) for i in r[:-1]]] = 1
        return match_matrix

    def _build_pymatch_obj(self) -> None:
        match_matrix_unique, unique_indices = np.unique(self.matching_matrix, axis=1, return_index=True)
        self._pymatch_obj = pymatching.Matching(match_matrix_unique, repetitions=self._rounds + 1)
        self._unique_match_indices = unique_indices

    @property
    def logical_vecs(self) -> np.ndarray:
        vecs = np.zeros((len(self._logical_observables), len(self._data_qubits)), dtype=np.uint8)
        for k in range(len(self._logical_observables)):
            vecs[k, np.isin(self._data_qubits, self._logical_observables[k])] = 1
        return vecs

    @property
    def distance(self) -> int:
        return self._distance

    @property
    def code_task(self) -> str:
        return self._code_task

    @property
    def params(self) -> CircuitParams:
        return self._params

    @property
    def rounds(self) -> int:
        return self._rounds

    def decode(self, z: np.ndarray) -> np.ndarray:
        decoded_frame = np.zeros(len(self.data_qubits), dtype=np.uint8)
        decoded_frame[self._unique_match_indices] = self._pymatch_obj.decode(z)
        return decoded_frame
