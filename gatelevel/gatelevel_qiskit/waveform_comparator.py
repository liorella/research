from deepdiff import DeepDiff
import numpy as np


def _traverse_dicts(waveforms, iter_function, node_function):
    if isinstance(waveforms, (dict, list)):

        if iter_function is not None:
            iter_function(waveforms)
        itr = iter(waveforms)
        for element in itr:
            if isinstance(waveforms, dict):
                _traverse_dicts(waveforms[element], iter_function, node_function)
            else:
                _traverse_dicts(element, iter_function, node_function)

    else:
        if node_function is not None:
            node_function(waveforms)


class WaveformComparator:
    def __init__(self, simulator_waveforms, circuit_waveforms):
        self._simulator_waveforms = simulator_waveforms
        self._circuit_waveforms = circuit_waveforms
        self._transform_literal_constant()
        self._phase_mod_2pi()
        self.diff = DeepDiff(simulator_waveforms, circuit_waveforms, significant_digits=5)
        self._transform_names()

    def __repr__(self):
        return repr(self.diff)

    def _phase_mod_2pi(self):
        def transform_func(element):
            if 'phase' in element:
                element['phase'] = element['phase'] % (2 * np.pi)
                if abs(element['phase'] - 2 * np.pi) < 1e-5:
                    element['phase'] = 0.0

        _traverse_dicts(self._circuit_waveforms, transform_func, None)
        _traverse_dicts(self._simulator_waveforms, transform_func, None)

    def _transform_literal_constant(self):
        def transform_func(element):
            if 'samples' in element:
                if element['samples']['type'] == 'constant':
                    element['samples']['values'] = [element['samples']['value']] * int(element['duration'])
                    element['samples'].pop('value')
                    element['samples']['type'] = 'literal'

        _traverse_dicts(self._simulator_waveforms, transform_func, None)

    def _transform_names(self):
        def trasform_func(element):
            if "old_value" in element:
                element["qua_sim"] = element["old_value"]
                element.pop("old_value")
            if "new_value" in element:
                element["waveform_sim"] = element["new_value"]
                element.pop("new_value")

        _traverse_dicts(self.diff, trasform_func, None)
