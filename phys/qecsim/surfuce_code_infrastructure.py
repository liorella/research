import abc
import dataclasses
from enum import Enum
from functools import reduce

import stim
import pymatching
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List
import networkx as nx


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


class SurgeryOrientation(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


class SurgeryOperation(Enum):
    ZZ = 0
    XZ = 1
    ZX = 2
    XX = 3


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


    def add_detectors_for_all_ancillas(self, circ, measurements: list):
        for ancilla in self._all_active_ancillas():
            if self.qubit_data(ancilla,measurements,-2) is None:
                circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1)])
            else:
                circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1), self.qubit_data(ancilla,measurements,-2)])

    def print_ancillas(self):
        print(np.flipud(self.ancilla_qubits.T))

    def print_data(self):
        print(np.flipud(self.data_qubits.T))

    @abc.abstractmethod
    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        pass


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
        for ancilla in ancillas_for_detectors:
            circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1)])

    def feedback_target(self, ancilla):
        (x, y) = np.where(self.ancilla_qubits == ancilla)
        x = x[0]
        y = y[0]
        d = self.dist
        d2 = (d + 1) / 2
        if y == 0:
            return self.data_qubits.T[0][0:x] if x < d2 else self.data_qubits.T[0][x:d]
        if x == 0:
            return self.data_qubits[0][0:y] if y < d2 else self.data_qubits[0][y:d]
        if (x + y) % 2 == 0:
            return self.data_qubits.T[y - 1][0:x] if x < d2 else self.data_qubits.T[y - 1][x:d]
        else:
            return self.data_qubits[x - 1][0:y] if y < d2 else self.data_qubits[x - 1][y:d]

    def add_initialization_feedback(self, circ, measurements: list, error_model: BaseErrorModel):
        if self.initial_state == InitialState.Z_PLUS or self.initial_state == InitialState.Z_MINUS:
            ancillas_for_feedback = self.ancilla_groups[0]
            command = "CZ"
        elif self.initial_state == InitialState.X_PLUS or self.initial_state == InitialState.X_MINUS:
            ancillas_for_feedback = self.ancilla_groups[1]
            command = "CX"
        for ancilla in ancillas_for_feedback:
            target_qubits = self.feedback_target(ancilla)
            for qubit in target_qubits:
                circ.append(command, [self.qubit_data(ancilla,measurements,-1), qubit])
            measurements[np.where(np.array(measurements) == ancilla)[0][0]] = -1
            error_model.generate_single_qubit_error(circ, target_qubits)

    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        if self.round == 1:
            self.add_surface_initialization_detectors(circ, measurements)
            self.add_initialization_feedback(circ, measurements, error_model)
        elif self.round > 1:
            self.add_detectors_for_all_ancillas(circ, measurements)

    def print_surface_name(self):
        print(self.data_qubits)


class LatticeSurgery(BaseSurface):

    def __init__(self, surface1: Surface, surface2: Surface, surgery_orientation: SurgeryOrientation):
        super().__init__(
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.HORIZONTAL else surface1.dist,
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.VERTICAL else surface1.dist
        )
        self.surface1 = surface1
        self.surface2 = surface2
        self.orientation = surgery_orientation
        if surface1.dist != surface2.dist:
            raise RuntimeError("Surfaces should be with the same dist")
        self.surgery_data_qubits = surface1.to_surgery_data_qubits[
            'T'] if surgery_orientation == SurgeryOrientation.VERTICAL else surface1.to_surgery_data_qubits['R']

    def _surgery_operation(self):
        return SurgeryOperation((self.surface1.orientation.value + self.surface2.orientation.value * 2) * (
                    1 - self.orientation.value) + self.orientation.value * ((1 - self.surface1.orientation.value) + (
                    1 - self.surface2.orientation.value) * 2))

    def _allocate_data_qubits(self):
        dist = self.surface1.dist
        self.data_qubits[0:dist, 0:dist] = self.surface1.data_qubits
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.data_qubits[(dist + 1):(2 * dist + 1), 0:dist] = self.surface2.data_qubits
            self.data_qubits[dist, 0:dist] = self.surgery_data_qubits
        else:
            self.data_qubits[0:dist, (dist + 1):(2 * dist + 1)] = self.surface2.data_qubits
            self.data_qubits[0:dist, dist] = self.surgery_data_qubits

    def _allocate_ancillas(self):
        self.ancilla_groups[0] = self.surface1.ancilla_groups[0].union(self.surface2.ancilla_groups[0])
        self.ancilla_groups[1] = self.surface1.ancilla_groups[1].union(self.surface2.ancilla_groups[1])
        dist = self.surface1.dist
        self.ancilla_qubits[0:dist + 1, 0:dist + 1] = self.surface1.ancilla_qubits
        op_val = self._surgery_operation().value
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.ancilla_qubits[dist + 1:, 0:dist + 1] = self.surface2.ancilla_qubits
            self.ancilla_groups[self.surface1.orientation.value].update(self.surface1.ancilla_qubits[-1, 1::2])
            if not (op_val % 3):
                self.ancilla_groups[self.surface2.orientation.value].update(self.surface2.ancilla_qubits[0, 0::2])
            else:
                self.ancilla_groups[1 + op_val].update(self.surface2.ancilla_qubits[0, 0::2])
                self.ancilla_groups[4 - op_val].update(self.surface2.ancilla_qubits[0, 1:-1:2])
                self.ancilla_groups[1 - self.surface2.orientation.value] -= set(self.surface2.ancilla_qubits[0, 1::2])
        elif self.orientation == SurgeryOrientation.VERTICAL:
            self.ancilla_qubits[0:dist + 1, dist + 1:] = self.surface2.ancilla_qubits
            self.ancilla_groups[1 - self.surface1.orientation.value].update(self.surface1.ancilla_qubits[0::2, -1])
            if not (op_val % 3):
                self.ancilla_groups[1 - self.surface2.orientation.value].update(self.surface2.ancilla_qubits[1::2, 0])
            else:
                self.ancilla_groups[3 + op_val].update(self.surface2.ancilla_qubits[1::2, 0])
                self.ancilla_groups[6 - op_val].update(self.surface2.ancilla_qubits[2::2, 0])
                self.ancilla_groups[self.surface2.orientation.value] -= set(self.surface2.ancilla_qubits[2::2, 0])

    def _allocate_to_surgery_data_qubits(self):
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.to_surgery_data_qubits['L'] = self.surface1.to_surgery_data_qubits['L']
            self.to_surgery_data_qubits['R'] = self.surface2.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['T'][0:self.surface1.width] = self.surface1.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['T'][self.surface1.width] = max(self.surface1.to_surgery_data_qubits['T']) + 1
            self.to_surgery_data_qubits['T'][self.surface1.width + 1:self.surface1.width + self.surface2.width + 1] = \
            self.surface2.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['B'][0:self.surface1.width] = self.surface1.to_surgery_data_qubits['B']
            self.to_surgery_data_qubits['B'][self.surface1.width] = max(
                self.surface1.to_surgery_data_qubits['T']) + 1 - 1000
            self.to_surgery_data_qubits['B'][self.surface1.width + 1:self.surface1.width + self.surface2.width + 1] = \
            self.surface2.to_surgery_data_qubits['B']
        else:
            self.to_surgery_data_qubits['B'] = self.surface1.to_surgery_data_qubits['B']
            self.to_surgery_data_qubits['T'] = self.surface2.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['R'][0:self.surface1.height] = self.surface1.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['R'][self.surface1.height] = max(self.surface1.to_surgery_data_qubits['T']) + 1
            self.to_surgery_data_qubits['R'][self.surface1.height + 1:self.surface1.dist + self.surface2.height + 1] = \
            self.surface2.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['L'][0:self.surface1.height] = self.surface1.to_surgery_data_qubits['L']
            self.to_surgery_data_qubits['L'][self.surface1.height] = max(
                self.surface1.to_surgery_data_qubits['T']) + 1 - 10000
            self.to_surgery_data_qubits['L'][self.surface1.height + 1:self.surface1.height + self.surface2.height + 1] = \
            self.surface2.to_surgery_data_qubits['L']

    def allocate_qubits(self, coord):
        self._allocate_data_qubits()
        self._allocate_ancillas()
        self._allocate_to_surgery_data_qubits()

    def initialize_surgery_data(self, circ:stim.Circuit, error_model: BaseErrorModel):
        circ.append("R", self.surgery_data_qubits)
        if not (self._surgery_operation().value % 2):
            circ.append("H", self.surgery_data_qubits)
        error_model.generate_single_qubit_error(circ, self.surgery_data_qubits)

    def observable_qubits(self):
        dist = self.surface1.dist
        return np.concatenate((self.ancilla_qubits[dist, 1::2], self.ancilla_qubits[dist + 1,
                                                                            0::2])).flatten() if self.orientation == SurgeryOrientation.HORIZONTAL else \
            np.concatenate((self.ancilla_qubits[0::2, dist], self.ancilla_qubits[1::2, dist + 1])).flatten()

    def add_surgery_initialization_detectors(self, circ:stim.Circuit, measurements: list):
        ancillas_for_detection = self._all_active_ancillas() - set(self.observable_qubits())
        for ancilla in ancillas_for_detection:
            circ.append("DETECTOR", [self.qubit_data(ancilla,measurements,-1), self.qubit_data(ancilla,measurements,-2)])

    def add_surgery_initialization_feedback(self,circ,measurements, error_model: BaseErrorModel):
        dist = self.surface1.dist
        operation=self._surgery_operation()
        for j in range(0, dist, 2):
            qubits = self.data_qubits[dist:2*dist+1,j] if self.orientation == SurgeryOrientation.HORIZONTAL else self.data_qubits[j,dist:2*dist+1]
            ancilla= self.ancilla_qubits[dist + 1, j] if self.orientation == SurgeryOrientation.HORIZONTAL else self.ancilla_qubits[j+1, dist + 1]
            ancilla2= self.ancilla_qubits[dist, j+1] if self.orientation == SurgeryOrientation.HORIZONTAL else self.ancilla_qubits[j, dist]
            for i, qubit in enumerate(qubits):
                if i == 0:
                    command = "CZ" if operation.value % 2 else "CX"
                    circ.append(command, [self.qubit_data(ancilla2, measurements, -1), qubit])
                else:
                    command = "CZ" if operation.value > 1 else "CX"
                    circ.append(command, [self.qubit_data(ancilla, measurements, -1), qubit])
                    circ.append(command, [self.qubit_data(ancilla2, measurements, -1), qubit])
            measurements[np.where(np.array(measurements) == ancilla)[0][0]] = -1
            measurements[np.where(np.array(measurements) == ancilla2)[0][0]] = -1
            error_model.generate_single_qubit_error(circ,qubits)

    def observable_data(self,measurements: list):
        observable_data = []
        for qubits in self.observable_qubits():
            observable_data.append(self.qubit_data(qubits, measurements, -1))
        return observable_data

    def add_observable(self, circ:stim.Circuit, measurements, observable):
        circ.append('OBSERVABLE_INCLUDE', self.observable_data(measurements), observable)


    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        if self.round == 1:
            self.add_surgery_initialization_detectors(circ, measurements)
            self.add_surgery_initialization_feedback(circ,measurements, error_model)
        else:
            self.add_detectors_for_all_ancillas(circ, measurements)

    def add_surgery_measurement_feedback(self, circ, measurements: list, error_model: BaseErrorModel):
        dist=self.surface1.dist
        control_qubits = self.surgery_data_qubits
        for j, qubit in enumerate(control_qubits):
            if self.orientation==SurgeryOrientation.HORIZONTAL:
               target_qubits = np.concatenate([self.data_qubits[dist - 1, 0:j], self.data_qubits[dist + 1, 0:j], [self.data_qubits[dist + (-1)**j, j]]])  if j < (dist + 1) / 2 \
                   else np.concatenate([self.data_qubits[dist-1,j+1:],self.data_qubits[dist+1,j+1:],[self.data_qubits[dist + (-1)**(j+1), j]]])
            else:
                target_qubits = np.concatenate([self.data_qubits[0:j,dist - 1], self.data_qubits[0:j,dist + 1],[self.data_qubits[j,dist + (-1)**(j+1)]]]) if j < (dist - 1) / 2 \
                    else np.concatenate([self.data_qubits[j+1:,dist-1],self.data_qubits[j+1:,dist+1],[self.data_qubits[j,dist + (-1)**j]]])
            for target in target_qubits:
                if target<qubit:
                    command= "CX" if self._surgery_operation().value % 2 else "CZ"
                else:
                    command = "CX" if self._surgery_operation().value > 1 else "CZ"
                circ.append(command, [self.qubit_data(qubit, measurements, -1), target])
            error_model.generate_single_qubit_error(circ,target_qubits)


    def measure_surgery_data(self, circ, measurements,error_model: BaseErrorModel):
        error_model.generate_measurement_qubit_error(circ, self.surgery_data_qubits)
        if not (self._surgery_operation().value % 2):
            circ.append("H", self.surgery_data_qubits)
        circ.append("M", self.surgery_data_qubits)
        measurements.extend(self.surgery_data_qubits)
        self.add_surgery_measurement_feedback(circ, measurements, error_model)



class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface], error_model: BaseErrorModel):
        self.surfaces = surfaces
        self.circ = stim.Circuit()
        self.surgeries: Dict[tuple, LatticeSurgery] = {}
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(coordinate)

        for coordinate, surface in surfaces.items():
            self._allocate_surgery(surface, coordinate, SurgeryOrientation.HORIZONTAL)
            self._allocate_surgery(surface, coordinate, SurgeryOrientation.VERTICAL)

        self.activated_surfaces: List[BaseSurface] = []
        self.measurements = []
        self.error_model = error_model
        self.observables = 0

    def _allocate_surgery(self, surface, coordinate, orientation: SurgeryOrientation):
        other_coord = (coordinate[0], coordinate[1] + 1) if orientation == SurgeryOrientation.VERTICAL else (
            coordinate[0] + 1, coordinate[1])
        if other_coord not in self.surfaces:
            return
        surgery = LatticeSurgery(surface, self.surfaces[other_coord], orientation)
        self.surgeries[coordinate, other_coord] = surgery

    def activate_surface(self, surface: BaseSurface):
        if isinstance(surface, Surface):
            self.activated_surfaces = [x for x in self.activated_surfaces if
                                       (isinstance(x, Surface) or (x.surface1 != surface and x.surface2 != surface))]
            self.activated_surfaces.append(surface)
        elif isinstance(surface, LatticeSurgery):
            self.activated_surfaces = [x for x in self.activated_surfaces if
                                       (isinstance(x, LatticeSurgery) or (
                                                   x != surface.surface1 and x != surface.surface2))]
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
            surface.add_detectors(self.circ, self.measurements, self.error_model)

    def initialize_surgery(self, coord0: tuple, coord1: tuple):
        surgery = self.surgeries[(coord0, coord1)]
        surgery.allocate_qubits(coord0)
        self.activate_surface(surgery)
        surgery.initialize_surgery_data(self.circ, self.error_model)

    def measure_surgery(self, coord0: tuple, coord1: tuple):
        surgery = self.surgeries[(coord0, coord1)]
        surgery.add_observable(self.circ, self.measurements, self.observables)
        self.observables += 1
        surgery.measure_surgery_data(self.circ, self.measurements, self.error_model)
        self.activate_surface(surgery.surface1)
        self.activate_surface(surgery.surface2)


##

error_model = ErrorModel(single_qubit_error=0.001, two_qubit_error=0.005, measurement_error=0.005)
#error_model = NoErrorModel()
d = 9
ex = Experiment({
    (0, 0): Surface(d),
    (0, 1): Surface(d),
    (1, 0): Surface(d)
}, error_model)
# ex.flip_surface_orientation((1,0))
# ex.flip_surface_orientation((0,0))

ex.initialize_surface((0, 0), InitialState.X_PLUS)
ex.initialize_surface((0, 1), InitialState.X_PLUS)
ex.initialize_surface((0, 1), InitialState.X_PLUS)

ex.stabilizer_round()
ex.stabilizer_round()
ex.stabilizer_round()


ex.initialize_surgery((0, 0), (0, 1))
ex.stabilizer_round()
ex.stabilizer_round()

ex.stabilizer_round()
ex.measure_surgery((0, 0), (0, 1))

# ex.stabilizer_round()
ex.stabilizer_round()
ex.stabilizer_round()
ex.stabilizer_round()

ex.measure_surface((0, 0), Basis.X_BASIS, apply_feedback= 0, feedback_basis= Basis.X_BASIS, feedback_surface=(0, 1))
ex.stabilizer_round()
ex.stabilizer_round()

ex.measure_surface((0, 1), Basis.Z_BASIS)
# ex.measure_surface((1, 0), MeasurementBasis.Z_BASIS)


##
model = ex.circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
num_shots = 10000
sampler = ex.circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
predicted_observables = matching.decode_batch(syndrome)
num_errors = sum(predicted_observables != actual_observables)
print(num_errors)

print(np.sum(actual_observables, 0))
print(np.sum(predicted_observables, 0))

##
E = matching.edges()  # edges and wieghts
G = matching.to_networkx()  # the documentation for networkX graph can be used
options = {
    "font_size": 10,
    "node_size": 200,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
plt.close()
nx.draw_networkx(G, with_labels=True, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0)
plt.axis("off")
plt.show()

## simulations

# error_model = ErrorModel(single_qubit_error=0.001, two_qubit_error=0.005, measurement_error=0.005)
error_model = NoErrorModel()
d_vec = range(3, 4, 2)
surgery_rounds = range(1, 2, 1)
num_shots = 10000
errors = np.zeros([len(d_vec), len(surgery_rounds), 4])
for j, d in enumerate(d_vec):
    for rounds in surgery_rounds:
        ex = Experiment({
            (0, 0): Surface(d),
            (1, 0): Surface(d),
            (0, 1): Surface(d)
        }, error_model)

        ex.initialize_surface((0, 0), InitialState.X_PLUS)
        ex.initialize_surface((1, 0), InitialState.X_MINUS)
        ex.initialize_surface((0, 1), InitialState.Z_PLUS)
        for i in range(d):
            ex.stabilizer_round()
        ex.initialize_surgery((0, 0), (1, 0))
        for i in range(rounds):
            ex.stabilizer_round()
        ex.measure_surgery((0, 0), (1, 0))
        for i in range(d):
            ex.stabilizer_round()
        ex.measure_surface((0, 0), Basis.X_BASIS)
        ex.measure_surface((1, 0), Basis.X_BASIS)
        ex.measure_surface((0, 1), Basis.Z_BASIS)

        model = ex.circ.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        sampler = ex.circ.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

        predicted_observables = matching.decode_batch(syndrome)
        errors[j, rounds - 1, :] = sum(predicted_observables != actual_observables)

print(errors)
##
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
labels = ["surgery", "q0", "q1", "q2"]

for j, ax in enumerate(axs.flat):
    for i in range(errors.shape[2]):
        ax.scatter(surgery_rounds, errors[j, :, i] / num_shots, label=labels[i], marker='o')
    ax.set_title(f'distance = {d_vec[j]}')
axs[-1, 0].set_xlabel('surgery rounds')
axs[-1, -1].set_xlabel('surgery rounds')
axs[-1, 0].set_ylabel('error probability')
axs[0, 0].set_ylabel('error probability')

axs[0, 0].legend()  # Adjust the location and number of columns

# Adjust the layout to prevent overlapping labels
plt.tight_layout()

# Display the plot
plt.show()
##

fig, ax = plt.subplots()

for j, d in enumerate(d_vec):
    ax.scatter(surgery_rounds, errors[j, :, 0] / num_shots, label=f'distance = {d}', marker='o')
    ax.set_xlabel('surgery rounds')
    ax.set_ylabel('error probability')
plt.legend()
plt.show()

## simulations for post-surgery rounds
error_model = ErrorModel(single_qubit_error=0.001, two_qubit_error=0.005, measurement_error=0.005)
# error_model = NoErrorModel()
d_vec = range(3, 11, 2)
post_surgery_rounds = range(1, 10, 1)
num_shots = 10000
post_surgery_errors = np.zeros([len(d_vec), len(post_surgery_rounds), 4])
for j, d in enumerate(d_vec):
    for rounds in post_surgery_rounds:
        ex = Experiment({
            (0, 0): Surface(d),
            (1, 0): Surface(d),
            (0, 1): Surface(d)
        }, error_model)

        ex.initialize_surface((0, 0), InitialState.X_PLUS)
        ex.initialize_surface((1, 0), InitialState.X_MINUS)
        ex.initialize_surface((0, 1), InitialState.Z_PLUS)
        for i in range(d):
            ex.stabilizer_round()
        ex.initialize_surgery((0, 0), (1, 0))
        for i in range(d):
            ex.stabilizer_round()
        ex.measure_surgery((0, 0), (1, 0))
        for i in range(rounds):
            ex.stabilizer_round()
        ex.measure_surface((0, 0), Basis.X_BASIS)
        ex.measure_surface((1, 0), Basis.X_BASIS)
        ex.measure_surface((0, 1), Basis.Z_BASIS)

        model = ex.circ.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        sampler = ex.circ.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        predicted_observables = matching.decode_batch(syndrome)
        post_surgery_errors[j, rounds - 1, :] = sum(predicted_observables != actual_observables)

##
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
labels = ["surgery", "q0", "q1", "q2"]

for j, ax in enumerate(axs.flat):
    for i in range(post_surgery_errors.shape[2]):
        ax.scatter(post_surgery_rounds, post_surgery_errors[j, :, i] / num_shots, label=labels[i], marker='o')
    ax.set_title(f'distance = {d_vec[j]}')
axs[-1, 0].set_xlabel('post surgery rounds')
axs[-1, -1].set_xlabel('post surgery rounds')
axs[-1, 0].set_ylabel('error probability')
axs[0, 0].set_ylabel('error probability')
axs[0, 0].legend()
plt.tight_layout()  # Adjust the layout to prevent overlapping labels
plt.show()  # Display the plot
##

fig, ax = plt.subplots()

for j, d in enumerate(d_vec):
    ax.scatter(post_surgery_rounds, post_surgery_errors[j, :, 0] / num_shots, label=f'distance = {d}', marker='o')
    ax.set_xlabel('post surgery rounds')
    ax.set_ylabel('error probability')
plt.legend()
plt.show()
