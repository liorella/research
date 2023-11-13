import abc
import dataclasses
from enum import Enum
from functools import reduce

import stim
import pymatching
import numpy as np
from typing import Dict, List


##
class TileOrder:
    order_z = ['NW', 'NE', 'SW', 'SE']
    order_ᴎ = ['NW', 'SW', 'NE', 'SE']


class SurfaceOrientation(Enum):
    Z_VERTICAL_X_HORIZONTAL = 1
    X_VERTICAL_Z_HORIZONTAL = 0


class InitialState(Enum):
    Z_PLUS = 0
    X_PLUS = 1
    Z_MINUS = 2
    X_MINUS = 3
    Y_PLUS = 4
    Y_MINUS = 5


class Basis(Enum):
    Z_BASIS = 0
    X_BASIS = 1


class BaseErrorModel(abc.ABC):
    @abc.abstractmethod
    def generate_single_qubit_error(self, circ, qubits):
        pass

    def generate_two_qubit_error(self, circ, qubits):
        pass

    def generate_measurement_qubit_error(self, circ, qubits):
        pass


class NoErrorModel(BaseErrorModel):
    def generate_single_qubit_error(self, circ, qubits):
        pass

    def generate_two_qubit_error(self, circ, qubits):
        pass

    def generate_measurement_qubit_error(self, circ, qubits):
        pass


@dataclasses.dataclass
class ErrorModel(BaseErrorModel):
    single_qubit_error: float
    two_qubit_error: float
    measurement_error: float

    def generate_single_qubit_error(self, circ, qubits):
        circ.append("DEPOLARIZE1", qubits, self.single_qubit_error)

    def generate_two_qubit_error(self, circ, qubits):
        circ.append("DEPOLARIZE2", qubits, self.two_qubit_error)

    def generate_measurement_qubit_error(self, circ, qubits):
        circ.append("X_ERROR", qubits, self.measurement_error)


class BaseSurface(abc.ABC):

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data_qubits = np.zeros((width, height), dtype=int)
        self.ancilla_qubits = np.zeros((width + 1, height + 1), dtype=int)
        self.ancilla_groups = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(),
                               5: set()}  # 0= X stabilizer, 1= Z stabilizer, 2=X_left Z_right, 3=Z_left X_right, 4=Z_top X_bottom, 5=X_top Z_bottom
        self.even_tiles_order = TileOrder.order_z
        self.round = 0
        self.to_surgery_data_qubits = {'R': np.zeros((height,), dtype=int),
                                       'L': np.zeros((height,), dtype=int),
                                       'T': np.zeros((width,), dtype=int),
                                       'B': np.zeros((width,), dtype=int)}

    @abc.abstractmethod
    def allocate_qubits(self, coord):
        pass

    def _all_active_ancillas(self):
        return reduce(lambda acc, x: acc.union(x), self.ancilla_groups.values(), set())

    def _get_target(self, ancilla_index,
                    direction):  # get ancilla and direction return corresponding data or none if no qubit
        if direction == 'SW':
            ret = ancilla_index[0] - 1, ancilla_index[1] - 1
        elif direction == 'NW':
            ret = ancilla_index[0] - 1, ancilla_index[1]
        elif direction == 'NE':
            ret = ancilla_index[0], ancilla_index[1]
        elif direction == 'SE':
            ret = ancilla_index[0], ancilla_index[1] - 1
        return None if ret[0] < 0 or ret[1] < 0 or ret[0] >= self.width or ret[1] >= self.height else ret

    def _get_ancilla_with_targets_and_op(self, epoch,
                                         stabilizer_group: int):  # gets direction of 2 qubit gate and which stabilizer_group (orientation independent), creates pair (source and target qubits)
        qubits = []
        operation = []
        my_ancillas = self.ancilla_groups[stabilizer_group]
        direction = self.even_tiles_order[epoch - 2]
        for ancilla in my_ancillas:
            loc = np.where(self.ancilla_qubits == ancilla)
            if (loc[0][0] + loc[1][0]) % 2 and (epoch == 3 or epoch == 4):
                direction = self.even_tiles_order[5 - epoch]
            ancilla_coord = np.where(self.ancilla_qubits == ancilla)
            target = self._get_target((ancilla_coord[0][0], ancilla_coord[1][0]), direction)
            if target is not None:
                qubits += ancilla, self.data_qubits[target]
            if stabilizer_group == 0 or (direction == 'NW' and (stabilizer_group == 2 or stabilizer_group == 5)) \
                    or (direction == 'NE' and (stabilizer_group == 3 or stabilizer_group == 5)) \
                    or (direction == 'SE' and (stabilizer_group == 3 or stabilizer_group == 4)) \
                    or (direction == 'SW' and (stabilizer_group == 2 or stabilizer_group == 4)):
                operation = "CX"
            else:
                operation = "CZ"
        return qubits, operation

    def _apply_two_qubit_gate_epoch(self, circ, epoch, error_model: BaseErrorModel):
        for ancilla_group in range(6):  # 2=X_left Z_right, 3=Z_left X_right, 4=Z_top X_bottom, 5=X_top Z_bottom
            [qubits, operation] = self._get_ancilla_with_targets_and_op(epoch, ancilla_group)
            if len(qubits):
                circ.append(operation, qubits)
                error_model.generate_two_qubit_error(circ, qubits)

    def stabilizer_round(self, circ, epoch: int, measurements: list, error_model: BaseErrorModel):
        ancillas = self._all_active_ancillas()
        if epoch == 0:
            circ.append("R", ancillas)
        elif epoch == 1:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch < 6:
            self._apply_two_qubit_gate_epoch(circ, epoch, error_model)
        elif epoch == 6:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas)
            circ.append("M", ancillas)
            measurements.extend(ancillas)
            self.round += 1

    def qubit_data(self, qubit, measurements,loc):
        if len(np.where(np.array(measurements) == qubit)[0])>-loc-1:
            return stim.target_rec((np.where(np.array(measurements) == qubit)[0] - len(measurements))[loc])
        else:
            return None


class Surface(BaseSurface):
    def __init__(self, dist: int):
        super().__init__(dist, dist)
        self.dist = dist
        self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        self.initial_state = InitialState.Z_PLUS

    def flip_orientation(self):
        if self.orientation.value:
            self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        else:
            self.orientation = SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL
        temp = self.ancilla_groups[0]
        self.ancilla_groups[0] = self.ancilla_groups[1]
        self.ancilla_groups[1] = temp

    def _allocate_to_surgery_data_qubits(self, name):
        dist = self.dist
        for i in range(2 * dist):
            if i < dist:
                self.to_surgery_data_qubits['R'][i % dist] = name
                self.to_surgery_data_qubits['L'][i % dist] = name - 10000
            else:
                self.to_surgery_data_qubits['T'][i % dist] = name
                self.to_surgery_data_qubits['B'][i % dist] = name - 1000
            name += 1
        return name

    def _allocate_ancillas(self, name):
        for i in range(self.dist + 1):
            for j in range(self.dist + 1):
                self.ancilla_qubits[i, j] = name
                if ((i + j) % 2 == 0 and self.orientation == SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL) or (
                        ((i + j) % 2 == 1) and (self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL)):
                    self.ancilla_groups[1].add(name)
                else:
                    self.ancilla_groups[0].add(name)
                name += 1
        to_remove = self.ancilla_qubits[0, 0::2].tolist() + self.ancilla_qubits[0::2, -1].tolist() + \
                    self.ancilla_qubits[1::2, 0].tolist() + self.ancilla_qubits[-1, 1::2].tolist()
        self.ancilla_groups[0] -= set(to_remove)
        self.ancilla_groups[1] -= set(to_remove)
        return name

    def _allocate_data_qubits(self, name):
        for i in range(self.dist):
            for j in range(self.dist):
                self.data_qubits[i, j] = name
                name += 1
        return name

    def allocate_qubits(self, coord):
        name = coord[0] * 10000 + coord[1] * 1000
        name = self._allocate_data_qubits(name)
        name = self._allocate_ancillas(name)
        name = self._allocate_to_surgery_data_qubits(name)

    def add_measurement_detectors(self, circ: stim.Circuit, basis: Basis, measurements: list):
        stabilizer_group = 0 if basis == Basis.X_BASIS else 1
        ancilla_target_list = []
        for epoch in [2, 3, 4, 5]:
            ancilla_target_list += self._get_ancilla_with_targets_and_op(epoch, stabilizer_group)[0]
        ancila_target_list = list(set(ancilla_target_list))
        ancillas = sorted(i for i in ancila_target_list if i > self.data_qubits[-1][-1])
        for ancilla in ancillas:
            locs = np.where(np.array(ancilla_target_list) == ancilla)[0]
            target = np.array(ancilla_target_list)[locs + 1]
            if len(target) == 2:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(target[0], measurements, -1), self.qubit_data(target[1], measurements, -1)])
            else:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(target[0], measurements, -1), self.qubit_data(target[1], measurements, -1),\
                             self.qubit_data(target[2], measurements, -1), self.qubit_data(target[3], measurements, -1)])
    def observable_data(self,measurements: list, basis: Basis):
        dist = self.dist
        observable_qubits = self.data_qubits[0:dist, 0] if self.orientation.value == basis.value else self.data_qubits[
                                                                                                      0, 0:dist]
        observable_data = []
        for qubits in observable_qubits.flatten():
            observable_data.append(self.qubit_data(qubits, measurements, -1))
        return observable_data
    def add_observable(self, circ: stim.Circuit, measurements: list, basis: Basis, observable_index: int):
        circ.append('OBSERVABLE_INCLUDE', self.observable_data(measurements,basis), observable_index)

    def apply_feedback(self, circ: stim.Circuit, observable_data, feedback: Basis, error_model: BaseErrorModel):
        dist = self.dist
        target_qubits = self.data_qubits[0:dist, 0] if self.orientation.value == feedback.value else self.data_qubits[
                                                                                                      0, 0:dist]
        for qubit in target_qubits:
            for data in observable_data:
                circ.append("CX", [data,qubit]) if feedback == Basis.X_BASIS else circ.append("CZ", [data, qubit])
        error_model.generate_single_qubit_error(circ, target_qubits)


    def surface_measurement(self, circ: stim.Circuit, basis: Basis, error_model: BaseErrorModel,
                            measurements: list):
        data_qubits = self.data_qubits.flatten()
        if basis == Basis.X_BASIS:
            circ.append('H', data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
            circ.append("Tick")
        error_model.generate_measurement_qubit_error(circ, data_qubits)
        circ.append('MZ', data_qubits)
        measurements.extend(data_qubits)
        self.round = 0
        self.add_measurement_detectors(circ, basis, measurements)

    def initialize_surface(self, circ, state: InitialState, error_model: BaseErrorModel):
        data_qubits = self.data_qubits.flatten()
        circ.append("R", data_qubits)
        if state == InitialState.Z_MINUS:
            circ.append("X", data_qubits)
            self.initial_state = InitialState.Z_MINUS
        elif state == InitialState.X_PLUS:
            circ.append("H", data_qubits)
            self.initial_state = InitialState.X_PLUS
        elif state == InitialState.X_MINUS:
            circ.append("X", data_qubits)
            circ.append("H", data_qubits)
            self.initial_state = InitialState.X_MINUS
        error_model.generate_single_qubit_error(circ, data_qubits)

    def add_surface_initialization_detectors(self, circ, measurements: list):
        if self.initial_state == InitialState.Z_PLUS or self.initial_state == InitialState.Z_MINUS:
            ancillas_for_detectors = self.ancilla_groups[1]
        elif self.initial_state == InitialState.X_PLUS or self.initial_state == InitialState.X_MINUS:
            ancillas_for_detectors = self.ancilla_groups[0]
        for ancilla in sorted(ancillas_for_detectors):
            if self.qubit_data(ancilla,measurements,-2) is None:
                circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1)])
            else:
                circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1), self.qubit_data(ancilla,measurements,-2)])

    def add_detectors(self, circ, measurements: list):
        self.add_surface_initialization_detectors(circ, measurements)

    def print_surface_name(self):
        print(self.data_qubits)


class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface], error_model: BaseErrorModel):
        self.surfaces = surfaces
        self.circ = stim.Circuit()
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(coordinate)

        self.activated_surfaces: List[BaseSurface] = []
        self.measurements = []
        self.error_model = error_model
        self.observables = 0

    def activate_surface(self, surface: BaseSurface):
        if isinstance(surface, Surface):
            self.activated_surfaces = [x for x in self.activated_surfaces if
                                       (isinstance(x, Surface) or (x.surface1 != surface and x.surface2 != surface))]
            self.activated_surfaces.append(surface)

    def __getitem__(self, coor):
        return self.surfaces[coor]

    def flip_surface_orientation(self, coor: tuple):
        self.surfaces[coor].flip_orientation()

    def measure_surface(self, coor: tuple, basis: Basis, apply_feedback=0, feedback_surface=(0, 0),
                        feedback_basis= Basis):
        self.surfaces[coor].surface_measurement(self.circ, basis, self.error_model, self.measurements)
        if not apply_feedback:
            self.surfaces[coor].add_observable(self.circ, self.measurements, basis, self.observables)
            self.observables += 1
        else:
            self.surfaces[feedback_surface].apply_feedback(self.circ,self.surfaces[coor].observable_data(self.measurements,basis), feedback_basis, self.error_model)
        self.activated_surfaces.remove(self.surfaces[coor])

    def initialize_surface(self, coor: tuple, state: InitialState):
        self.activate_surface(self.surfaces[coor])
        self.surfaces[coor].initialize_surface(self.circ, state, self.error_model)

    def stabilizer_round(self):
        for epoch in range(8):
            for surface in self.activated_surfaces:
                surface.stabilizer_round(self.circ, epoch, self.measurements, self.error_model)
            self.circ.append("TICK")
        for surface in self.activated_surfaces:
            surface.add_detectors(self.circ, self.measurements)



## experiment
p=0.01
error_model = ErrorModel(single_qubit_error=p, two_qubit_error=p, measurement_error=p)
d = 3
N = 4 # number of rounds
ex = Experiment({
    (0, 0): Surface(d),
}, error_model) #defining a single surface

ex.initialize_surface((0, 0), InitialState.Z_PLUS)
for _ in range(N):
    ex.stabilizer_round()

ex.measure_surface((0, 0), Basis.Z_BASIS)

##
model = ex.circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
num_shots = 10000
sampler = ex.circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
predicted_pymatching_observables = matching.decode_batch(syndrome)
pymatching_errors = sum(predicted_pymatching_observables != actual_observables)
print(pymatching_errors)

## reconstructing syndromes into the fusion_blossom shape
def expanding_matrix(arr, d):
    new_length = int(len(arr) + len(arr)/(d**2-1)*4*((d-1)/2+1))
    mapping_matrix = np.zeros((new_length,len(arr)))
    j=0
    for i in range(new_length):
        if (i - (d-1)/2)%(d+1) and (i - (d-1)/2-1)%(d+1):
            mapping_matrix[i][j] = 1
            j+=1
    return mapping_matrix

new_len = int(len(syndrome[0]) + len(syndrome[0])/(d**2-1)*4*((d-1)/2+1))
expanded_syndromes = np.zeros((num_shots, new_len))
mapping_mat=expanding_matrix(syndrome[0], d)
for i in range(num_shots):
    expanded_syndromes[i] = np.dot(mapping_mat,syndrome[i])

#now you can take the extended syndrome matrix to the fusion blossom (include num_shots)
## listing all the matchings that actually flip the logical pbservable
highlighted_edges=[0,4,6,7] #this is an example of a fusion blossom output of one shot
def was_flip(highlighted_edges, d):
    flip=0
    for i in range(len(highlighted_edges)):
        index=highlighted_edges[i]%(d**2+((d-1)/2+1)*d+1)
        if (index%d==0) and index<d**2:
            flip=flip^1
    return flip
##
fusion_blossom_errors=0
for i in range(num_shots):
    predicted_fusion_blossom_observables = was_flip(highlighted_edges[i],d)
    fusion_blossom_errors += predicted_fusion_blossom_observables != actual_observables[i]
print(fusion_blossom_errors)

-