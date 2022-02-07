from itertools import takewhile, dropwhile

import networkx as nx
import numpy as np
import stim


def _get_records(circ: stim.Circuit):
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
    Generates a context for decoding and running error correction sequences on generated stim circuits
    """

    def __init__(self, circuit: stim.Circuit):
        self._circuit = circuit

        measures = _get_records(self._circuit)
        self._match_indices = []

        # this assumes that the reversed structure of the generated program is logical observables and then
        # detectors with parities on the data qubits
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

    @property
    def ancillas(self):
        return self._ancillas

    @property
    def data_qubits(self):
        return self._data_qubits

    @property
    def matching_matrix(self) -> np.ndarray:
        match_matrix = np.zeros((len(self._ancillas), len(self._data_qubits)), dtype=np.uint8)
        for r in self._match_indices:
            match_matrix[self._ancillas.index(r[-1]), [self._data_qubits.index(i) for i in r[:-1]]] = 1
        return match_matrix

    @property
    def matching_graph_nx(self) -> nx.Graph:
        # todo
        raise NotImplementedError()
