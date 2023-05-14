import abc
import dataclasses
from enum import Enum

import stim
import pymatching
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List
import networkx as nx




##

class SurfaceOrientation(Enum):
    Z_VERTICAL_X_HORIZONTAL = 1
    X_VERTICAL_Z_HORIZONTAL = 0

class MeasurementBasis(Enum):
    Z_BASIS = 0
    X_BASIS = 1

class initialState(Enum):
    Z_PLUS = 0
    X_PLUS = 1
    Z_MINUS = 2
    X_MINUS = 3
    Y_PLUS = 4
    T_STATE = 5
class SurgeryOrientation(Enum):
    VERTICAL = 0
    HORIZONTAL = 1

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
        self.ancilla_qubits = {'L': np.zeros(((height - 1),), dtype=int),
                               'R': np.zeros(((height - 1),), dtype=int),
                               'T': np.zeros(((width - 1),), dtype=int),
                               'B': np.zeros(((width - 1),), dtype=int),
                               'C': np.zeros((width - 1, height - 1), dtype=int)}
        self.ancilla_groups = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set()} #0= X stabilizer, 1= Z stabilizer, 2=X_left Z_right, 3=Z_left X_right, 4=Z_top X_bottom, 5=X_top Z_bottom

    @abc.abstractmethod
    def allocate_qubits(self, coord):
        pass

    def _all_ancillas(self):
        res = []
        for a_list in self.ancilla_qubits.values():
            for a in a_list.flatten():
                if a >= 0:
                    res.append(a)
        return res

    def _get_target(self, ancilla_direction, ancilla_index, direction): #get ancilla and direction return corresponding data or none if no qubit
        if ancilla_direction == 'L':
            ancilla_index = (-1, ancilla_index)
        elif ancilla_direction == 'R':
            ancilla_index = (self.width-1, ancilla_index)
        elif ancilla_direction == 'T':
            ancilla_index = (ancilla_index, self.height-1)
        elif ancilla_direction == 'B':
            ancilla_index = (ancilla_index, -1)
        ret = None
        if direction == 'SW':
            ret = ancilla_index[0], ancilla_index[1]
        elif direction == 'NW':
            ret = ancilla_index[0], ancilla_index[1]+1
        elif direction == 'NE':
            ret = ancilla_index[0]+1, ancilla_index[1]+1
        elif direction == 'SE':
            ret = ancilla_index[0]+1, ancilla_index[1]
        return None if ret[0] < 0 or ret[1] < 0 or ret[0] >= self.width or ret[1] >= self.height else ret

    def _get_ancilla_with_targets(self, target_direction, stabilizer_group: int): # gets direction of 2 qubit gate and which stabilizer_group (orientation independent), creates pair (source and target qubits)
        ops = []
        my_ancillas = self.ancilla_groups[stabilizer_group]
        for direction in ['L', 'R', 'T', 'B']:
            for i, ancilla in enumerate(self.ancilla_qubits[direction]):
                if ancilla >= 0 and ancilla in my_ancillas:
                    target = self._get_target(direction, i, target_direction)
                    if target is not None:
                        ops += ancilla, self.data_qubits[target]

        for i in range(self.width-1):
            for j in range(self.height-1):
                ancilla = self.ancilla_qubits['C'][i, j]
                if ancilla in my_ancillas:
                    target = self._get_target('C', (i, j), target_direction)
                    if target is not None:
                        ops += ancilla, self.data_qubits[target]
        return ops

    def _apply_two_qubit_gate_epoch(self, circ, direction, error_model: BaseErrorModel):
        for ancilla_group in [0,1]: #need to add the other groups for XZ surgery
            op = self._get_ancilla_with_targets(direction, ancilla_group)
            if ancilla_group == 0:
                circ.append("CX", op)
            elif ancilla_group == 1:
                circ.append("CZ", op)
            error_model.generate_two_qubit_error(circ, op)



    def stabilizer_round(self, circ, epoch: int, measurements: list, error_model: BaseErrorModel):
        ancillas = self._all_ancillas()
        if epoch == 0:
            circ.append("R", ancillas)
        elif epoch == 1:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 2:
            self._apply_two_qubit_gate_epoch(circ, 'SE', error_model)
        elif epoch == 3:
            self._apply_two_qubit_gate_epoch(circ, 'SW', error_model)
        elif epoch == 4:
            self._apply_two_qubit_gate_epoch(circ, 'NE', error_model)
        elif epoch == 5:
            self._apply_two_qubit_gate_epoch(circ, 'NW', error_model)
        elif epoch == 6:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas)
            circ.append("M", ancillas)
            measurements.extend(ancillas)
            for ancilla in ancillas:
                occ = np.where(np.array(measurements) == ancilla)[0]-len(measurements)
                if len(occ) >= 2:
                    circ.append("DETECTOR", [stim.target_rec(occ[-1]), stim.target_rec(occ[-2])])
                else:
                    circ.append("DETECTOR", [stim.target_rec(occ[-1])])


class Surface(BaseSurface):
    def __init__(self, dist: int):
        super().__init__(dist, dist)
        self.dist = dist
        self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL

    def flip_orientation(self):
        if self.orientation.value:
            self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        else:
            self.orientation = SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL
        temp=self.ancilla_groups[0]
        self.ancilla_groups[0] = self.ancilla_groups[1]
        self.ancilla_groups[1] = temp

    def _allocate_boundary_ancillas(self, coord, direction: str, name):
        for i in range(self.dist-1):
            if ((direction == 'L' or direction == 'T') and (coord[0] + coord[1] + i) % 2 == 0) or \
               ((direction == 'B' or direction == 'R') and (coord[0] + coord[1] + i) % 2 != 0):
                self.ancilla_qubits[direction][i] = name
                if ((direction == 'L' or direction == 'R') and self.orientation == SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL) or ((direction == 'T' or direction == 'B') and self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL):
                    self.ancilla_groups[0].add(name)
                else:
                    self.ancilla_groups[1].add(name)
                name += 1
            else:
                self.ancilla_qubits[direction][i] = -1
        return name

    def _allocate_central_ancillas(self, coord, name):
        for i in range(self.dist-1):
            for j in range(self.dist-1):
                self.ancilla_qubits['C'][i, j] = name
                if ((coord[0] + coord[1] + i + j) % 2 == 0 and self.orientation == SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL) or (((coord[0] + coord[1] + i + j) % 2 == 1) and (self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL)):
                    self.ancilla_groups[1].add(name)
                else:
                    self.ancilla_groups[0].add(name)
                name += 1

    def _allocate_data_qubits(self, name):
        for i in range(self.dist):
            for j in range(self.dist):
                self.data_qubits[i, j] = name
                name += 1
        return name

    def allocate_qubits(self, coord):
        name = coord[0] * 10000 + coord[1] * 1000
        name = self._allocate_data_qubits(name)

        for direction in ['L', 'R', 'T', 'B']:
            name = self._allocate_boundary_ancillas(coord, direction, name)

        self._allocate_central_ancillas(coord, name)

    def _add_measurement_detectors(self, circ: stim.Circuit, basis: MeasurementBasis, measurements: list):
        stabilizer_group = 0 if basis == MeasurementBasis.X_BASIS else 1
        ancilla_target_list=[]
        for direction in ['NW', 'NE', 'SE', 'SW']:
            ancilla_target_list += self._get_ancilla_with_targets(direction, stabilizer_group)
        ancila_target_list=list(set(ancilla_target_list))
        ancillas=sorted(i for i in ancila_target_list if i>self.data_qubits[-1][-1])
        for ancilla in ancillas:
            locs = np.where(np.array(ancilla_target_list) == ancilla)[0]
            target=np.array(ancilla_target_list)[locs+1]
            ancilla_loc = (np.where(np.array(measurements) == ancilla)[0] - len(measurements))[-1]
            data_loc=[]
            data_loc.append(np.where(np.array(measurements) == target[0])[0][-1] - len(measurements))
            data_loc.append(np.where(np.array(measurements) == target[1])[0][-1] - len(measurements))
            if len(target) == 2:
                circ.append("DETECTOR", [stim.target_rec(ancilla_loc), stim.target_rec(data_loc[0]), stim.target_rec(data_loc[1])])
            else:
                data_loc.append(np.where(np.array(measurements) == target[2])[0][-1] - len(measurements))
                data_loc.append(np.where(np.array(measurements) == target[3])[0][-1] - len(measurements))
                circ.append("DETECTOR", [stim.target_rec(ancilla_loc), stim.target_rec(data_loc[0]), stim.target_rec(data_loc[1]), stim.target_rec(data_loc[2]), stim.target_rec(data_loc[3])])

    def _add_observable(self, circ: stim.Circuit, basis: MeasurementBasis, observable_index: int):
        observable_qubits=[]
        for j in range(self.dist):
            observable_qubits.append(stim.target_rec(- j-1)) if self.orientation.value != basis.value else observable_qubits.append(stim.target_rec(- j*self.dist-1))
        circ.append('OBSERVABLE_INCLUDE', observable_qubits, observable_index)

    def measurement(self, circ: stim.Circuit, basis: MeasurementBasis, error_model: BaseErrorModel, measurements: list, observable_index: int):
        data_qubits=self.data_qubits.flatten()
        if basis == MeasurementBasis.X_BASIS:
            circ.append('H', data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
            circ.append("Tick")
        error_model.generate_measurement_qubit_error(circ, data_qubits)
        circ.append('MZ', data_qubits)
        measurements.extend(data_qubits)
        self._add_measurement_detectors(circ, basis, measurements)
        self._add_observable(circ, basis, observable_index)


    def initialize(self, circ, state: initialState, epoch, error_model: BaseErrorModel, measurements):
        data_qubits=self.data_qubits.flatten()
        if epoch == 0:
            circ.append("R", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
        elif epoch == 1:
            if state == initialState.Z_MINUS:
                circ.append("X", data_qubits)
                error_model.generate_single_qubit_error(circ, data_qubits)
            elif state == initialState.X_PLUS:
                circ.append("H", data_qubits)
                error_model.generate_single_qubit_error(circ, data_qubits)
            elif state == initialState.X_MINUS:
                circ.append("X", data_qubits)
                error_model.generate_single_qubit_error(circ, data_qubits)
        elif (epoch == 2) and (state == initialState.X_MINUS):
            circ.append("H", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
        elif epoch < 10:
            self.stabilizer_round(circ, epoch-3, measurements, error_model)
        elif epoch == 10:
            error_model.generate_measurement_qubit_error(circ, self._all_ancillas())
            circ.append("M", self._all_ancillas())
            measurements.extend(self._all_ancillas())
            if state == initialState.Z_PLUS or state == initialState.Z_MINUS:
                ancillas_for_detectors = self.ancilla_groups[1]
            elif state == initialState.X_PLUS or state == initialState.X_MINUS:
                ancillas_for_detectors = self.ancilla_groups[0]
            for ancilla in ancillas_for_detectors:
                occ = np.where(np.array(measurements) == ancilla)[0] - len(measurements)
                circ.append("DETECTOR", [stim.target_rec(occ)])

    def print_surface_name(self):
        print(self.data_qubits)


class LatticeSurgery(BaseSurface):

    def __init__(self, surface1: Surface, surface2: Surface, surgery_orientation: SurgeryOrientation, first_surface_coords_parity: int):
        super().__init__(
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.HORIZONTAL else surface1.dist,
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.VERTICAL else surface1.dist
        )
        self.surface1 = surface1
        self.surface2 = surface2
        self.orientation = surgery_orientation
        self.first_surface_coords_parity = first_surface_coords_parity
        if surface1.dist != surface2.dist:
            raise RuntimeError("Surfaces should be with the same dist")
        self.surgery_data_qubits=set()
        self.surgery_ancilla_qubits = set()


    def _allocate_data_qubits(self):
        dist = self.surface1.dist
        last_qubit_of_surface1=self.surface1._all_ancillas()[-1]
        self.data_qubits[0:dist, 0:dist] = self.surface1.data_qubits
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.data_qubits[(dist + 1):(2 * dist + 1), 0:dist] = self.surface2.data_qubits
            self.data_qubits[dist, 0:dist] = range(last_qubit_of_surface1+1, last_qubit_of_surface1+1+dist)
            self.surgery_data_qubits.add(range(last_qubit_of_surface1+1, last_qubit_of_surface1+1+dist))
        else:
            self.data_qubits[0:dist, (dist + 1):(2 * dist + 1)] = self.surface2.data_qubits
            self.data_qubits[0:dist, dist] = range(last_qubit_of_surface1+1+dist, last_qubit_of_surface1+1+2*dist)
            self.surgery_data_qubits.add(range(last_qubit_of_surface1+1+dist, last_qubit_of_surface1+1+2*dist))

    # numbering: squared surface has d^2-1, the next d are the data for the horizonal surgery, the next d are the data for the vertical surgery,
    # the next 2 are for (surgery-to-the-left-or-bottom and surgery-to-the-right-or-top) if first_surface_coords_parity=0, else (surgery-to-the-right-or-bottom surgery-to-the-left-or-top)
    # the next 4*(d-1)/2 relate to the previous "-1" ancilla qubits on the boundary of the original surfuces, that are needed for the surgery
    def _find_which_ancilla_group(self, qubit_name):
        for ancilla_group in self.ancilla_groups:
            if np.where(qubit_name == self.ancilla_groups[ancilla_group]):
                return ancilla_group

    def _allocate_central_ancillas(self): #need to be modified for X-Z surgery
        self.ancilla_groups[0].add(self.surface1.ancilla_groups[0]+self.surface2.ancilla_groups[0])
        self.ancilla_groups[1].add(self.surface1.ancilla_groups[1] + self.surface2.ancilla_groups[1])
        dist = self.surface1.dist
        self.ancilla_qubits['C'][0:dist-1, 0:dist-1] = self.surface1.ancilla_qubits['C']
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.ancilla_qubits['C'][dist+1:2*dist,0:dist-1] = self.surface2.ancilla_qubits['C']
            self.ancilla_qubits['C'][dist - 1, 0:dist-1] = self.surface1.ancilla_qubits['R']
            next_ancilla_surface1=self.surface1._all_ancillas()[-1] + 2 * dist + 3 + int((dist - 1) / 2)
            next_ancilla_qubits1=range(next_ancilla_surface1, next_ancilla_surface1 + int((dist - 1)/ 2))
            self.ancilla_qubits['C'][dist - 1, np.where(self.surface1.ancilla_qubits['R'] == -1)] = next_ancilla_qubits1
            group1=(self._find_which_ancilla_group(np.where(self.surface1.ancilla_qubits['R'] != -1)[-1])+1)%2
            self.ancilla_qubits['C'][dist, 0:dist - 1] = self.surface2.ancilla_qubits['L']
            next_ancilla_surface2 = self.surface2._all_ancillas()[-1] + 2 * dist + 3
            next_ancilla_qubits2 = range(next_ancilla_surface2, next_ancilla_surface2 + int((dist - 1)/ 2))
            self.ancilla_qubits['C'][dist, np.where(self.surface2.ancilla_qubits['L'] == -1)] = next_ancilla_qubits2
            group2=(self._find_which_ancilla_group(np.where(self.surface2.ancilla_qubits['L'] != -1)[-1])+1)%2
        else:
            self.ancilla_qubits['C'][0:dist - 1, dist + 1:2 * dist] = self.surface2.ancilla_qubits['C']
            self.ancilla_qubits['C'][0:dist-1, dist - 1] = self.surface1.ancilla_qubits['T']
            next_ancilla_surface1 = self.surface1._all_ancillas()[-1] + 2 * dist + 3 + 2*int((dist - 1) / 2)
            next_ancilla_qubits1=range(next_ancilla_surface1, next_ancilla_surface1 + int((dist - 1)/ 2))
            self.ancilla_qubits['C'][np.where(self.surface1.ancilla_qubits['T'] == -1),dist - 1] = next_ancilla_qubits1
            group1 = (self._find_which_ancilla_group(np.where(self.surface1.ancilla_qubits['T'] != -1)[-1]) + 1) % 2
            self.ancilla_qubits['C'][0:dist - 1,dist] = self.surface2.ancilla_qubits['B']
            next_ancilla_surface2 = self.surface2._all_ancillas()[-1] + 2 * dist + 3+3*int((dist - 1) / 2)
            next_ancilla_qubits2 = range(next_ancilla_surface2, next_ancilla_surface2 + int((dist - 1)/ 2))
            self.ancilla_qubits['C'][np.where(self.surface2.ancilla_qubits['B'] == -1),dist] = next_ancilla_qubits2
            group2 = (self._find_which_ancilla_group(np.where(self.surface2.ancilla_qubits['B'] != -1)[-1]) + 1) % 2
        self.ancilla_groups[group1].add(next_ancilla_qubits1)
        self.surgery_ancilla_qubits.add(next_ancilla_qubits1)
        self.ancilla_groups[group2].add(next_ancilla_qubits2)
        self.surgery_ancilla_qubits.add(next_ancilla_qubits2)


    def _allocate_boundary_ancillas(self):
        next_ancilla_surface1=self.surface1._all_ancillas()[-1]+2*self.surface1.dist+1
        next_ancilla_surface2=self.surface2._all_ancillas()[-1]+2*self.surface2.dist+1
        parity=self.first_surface_coords_parity
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.ancilla_qubits['L'] = self.surface1.ancilla_qubits['L']
            self.ancilla_qubits['R'] = self.surface2.ancilla_qubits['R']
            if (1-parity):#(0,0) (2,0)...
                self.ancilla_qubits['T'] = np.concatenate((self.surface1.ancilla_qubits['T'], [next_ancilla_surface1+1], [next_ancilla_surface2+1], self.surface2.ancilla_qubits['T']))
                self.ancilla_qubits['B'] = np.concatenate((self.surface1.ancilla_qubits['B'], [-1], [-1], self.surface2.ancilla_qubits['B']))
            else:
                self.ancilla_qubits['T'] = np.concatenate(
                    (self.surface1.ancilla_qubits['T'], [-1], [-1], self.surface2.ancilla_qubits['T']))
                self.ancilla_qubits['B'] = np.concatenate((self.surface1.ancilla_qubits['B'], [next_ancilla_surface1], [next_ancilla_surface2], self.surface2.ancilla_qubits['B']))
            self.ancilla_groups[self._find_which_ancilla_group(np.where(self.surface2.ancilla_qubits['B'] != -1)[-1])].add([next_ancilla_surface1+parity,next_ancilla_surface2+parity]) # need to modify for XZ surgery
        else:
            self.ancilla_qubits['B'] = self.surface1.ancilla_qubits['B']
            self.ancilla_qubits['T'] = self.surface2.ancilla_qubits['T']
            if (1-parity):#(0,0) (2,0)...
                self.ancilla_qubits['R'] = np.concatenate((self.surface1.ancilla_qubits['R'], [next_ancilla_surface1+1], [next_ancilla_surface2+1], self.surface2.ancilla_qubits['R']))
                self.ancilla_qubits['L'] = np.concatenate((self.surface1.ancilla_qubits['L'], [-1], [-1], self.surface2.ancilla_qubits['L']))
            else:
                self.ancilla_qubits['R'] = np.concatenate(
                    (self.surface1.ancilla_qubits['R'], [-1], [-1], self.surface2.ancilla_qubits['R']))
                self.ancilla_qubits['L'] = np.concatenate((self.surface1.ancilla_qubits['L'], [next_ancilla_surface1], [next_ancilla_surface2], self.surface2.ancilla_qubits['L']))
            self.ancilla_groups[self._find_which_ancilla_group(np.where(self.surface2.ancilla_qubits['L'] != -1)[-1])].add([next_ancilla_surface1+parity,next_ancilla_surface2+parity]) # need to modify for XZ surgery
        self.surgery_ancilla_qubits.add([next_ancilla_surface1+parity,next_ancilla_surface2+parity])

    def allocate_qubits(self):
       self._allocate_data_qubits()
       self._allocate_central_ancillas()
       self._allocate_boundary_ancillas()

    def surgery_operation(self, circ, error_model: BaseErrorModel):

        # initialize surgery data qubits
        # d stabilizer rounds to the combined surface
        # measurement

        pass









class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface], error_model: BaseErrorModel):
        self.circ = stim.Circuit()
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(coordinate)
        self.surfaces = surfaces
        self.measurements = []
        self.error_model = error_model

    def __getitem__(self, coor):
        return self.surfaces[coor]

    def flip_surface_orientation(self, coor: tuple):
        self.surfaces[coor].flip_orientation()

    def measure_surface(self, coor: tuple, basis: MeasurementBasis, observable_index):
        self.surfaces[coor].measurement(self.circ, basis, self.error_model, self.measurements, observable_index)
    def Initialize_surfaces(self, states: tuple):
        for epoch in range(11):
            for i, surface in enumerate(self.surfaces.values()):
                surface.initialize(self.circ, states[i], epoch, self.error_model, self.measurements)
            self.circ.append("TICK")


    def stabilizer_round(self):
        for epoch in range(8):
            for surface in self.surfaces.values():
                surface.stabilizer_round(self.circ, epoch, self.measurements, self.error_model)
            self.circ.append("TICK")

    def lattice_surgury(self, coor1: tuple, coor2: tuple):


        if abs(coor1[0]-coor2[0]) == 1: #the surgery is in the x-axes (between the vertical edges logical operators)
            Surgeryorientation=SurgeryOrientation.HORIZONTAL
        elif abs(coor1[1]-coor2[1]) == 1: #the surgery is in the y-axes (between the horizontal edges logical operators)
            Surgeryorientation = SurgeryOrientation.VERTICAL
        LatticeSurgery.__init__(self.surfaces[coor1], self.surfaces[coor2], Surgeryorientation, (coor1[0]+coor1[1])%2)





##
error_model = ErrorModel(single_qubit_error=0.01, two_qubit_error=0.01, measurement_error=0.05)
#error_model = NoErrorModel()
d=3
ex = Experiment({
    (0,0): Surface(d),
    (1,0): Surface(d)
    # (0,1): Surface(d)
}, error_model)


##
ex[0, 0].flip_orientation()

ex.Initialize_surfaces([initialState.X_PLUS, initialState.Z_PLUS])

# ex.stabilizer_round()
# ex.stabilizer_round()
# ex.stabilizer_round()
# ex.stabilizer_round()
# ex.lattice_surgury((0, 0),(1, 0))
ex.measure_surface((0, 0), MeasurementBasis.X_BASIS, observable_index=0)
ex.measure_surface((1, 0), MeasurementBasis.Z_BASIS, observable_index=1)

 #print(ex.circ)
# ex.circ.diagram()

##
model = ex.circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

sampler = ex.circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)

print(sum(actual_observables))


##
E=matching.edges() # edges and wieghtsD
G=matching.to_networkx() #the documentation for networkX graph can be used
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

##
# ex.flip_surface_orientation((0,0))


##
ex.measure_surface((0, 0), MeasurementBasis.X_BASIS)
ex.measure_surface((0,1), MeasurementBasis.X_BASIS)












## dividing into ticks

circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                 distance=3,
                                 rounds=4)

ticks_counter=0

for inst in circuit:
    # print(inst)
    if inst == stim.CircuitInstruction('TICK', [], []):
        ticks_counter+=1
        # print(2)
    if isinstance(inst, stim.CircuitRepeatBlock):
        for j in range(inst.repeat_count):
            print(j)
            for inner_inst in inst.body_copy():
                if inner_inst == stim.CircuitInstruction('TICK', [], []):
                    ticks_counter+=1
print(ticks_counter)
            # for circ_segment2 in to_measure_segments(inst.body_copy()):
            #     circ_segment += circ_segment2
            #     if circ_segment[-1].name in measure_instructions:
            #         yield circ_segment
            #         circ_segment = stim.Circuit()


##
Circuit2 = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
TICK
H 9 10 14 15
TICK
CX 12 3 13 1 16 5 8 10 2 14 4 15
TICK
CX 12 6 13 4 16 8 1 9 5 14 7 15
TICK
CX 11 5 13 3 16 7 0 9 4 14 6 15
TICK
CX 11 2 13 0 16 4 7 10 1 14 3 15
TICK
H 9 10 14 15
TICK
M 9 10 11 12 13 14 15 16
TICK
DETECTOR rec[-6]
DETECTOR rec[-5]
DETECTOR rec[-4]
DETECTOR rec[-1]
M 0 1 2 3 4 5 6 7 8
DETECTOR rec[-15] rec[-7] rec[-4]
DETECTOR rec[-14] rec[-6] rec[-3]
DETECTOR rec[-13] rec[-9] rec[-8] rec[-6] rec[-5]
DETECTOR rec[-10] rec[-5] rec[-4] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
''')
model2 = Circuit2.detector_error_model(decompose_errors=True)
print(Circuit2)
Circuit2.diagram()
sampler2 = Circuit2.compile_detector_sampler()
syndrome2, actual_observables2 = sampler2.sample(shots=1000, separate_observables=True)

##
Circuit2 = stim.Circuit('''
R 1 3 5 8 10 12 15 17 19 2 9 11 13 14 16 18 25
TICK
H 2 11 16 25
TICK
CX 2 3 16 17 11 12 15 14 10 9 19 18
TICK
CX 2 1 16 15 11 10 8 14 3 9 12 18
TICK
CX 16 10 11 5 25 19 8 9 17 18 12 13
TICK
CX 16 8 11 3 25 17 1 9 10 18 5 13
TICK
H 2 11 16 25
TICK
MR 2 9 11 13 14 16 18 25
DETECTOR rec[-4]
DETECTOR rec[-7]
DETECTOR rec[-2]
DETECTOR rec[-5]
M 1 3 5 8 10 12 15 17 19
DETECTOR rec[-3] rec[-6] rec[-13]
DETECTOR rec[-5] rec[-6] rec[-8] rec[-9] rec[-16]
DETECTOR rec[-1] rec[-2] rec[-4] rec[-5] rec[-11]
DETECTOR rec[-4] rec[-7] rec[-14]
OBSERVABLE_INCLUDE(0) rec[-7] rec[-8] rec[-9]
''')
Circuit2.diagram()

model2 = Circuit2.detector_error_model(decompose_errors=True)


sampler2 = Circuit2.compile_detector_sampler()
syndrome2, actual_observables2 = sampler2.sample(shots=1000, separate_observables=True)
##Also len(stim.Circuit.shortest_graphlike_error()) will be too low if you forgot a detector somewhere, or didn't do enough rounds of stitching to make the measurement result reliable.

def count_determined_measurements_in_circuit(circuit: stim.Circuit) -> int: # should equal the number of detectors plus the number of observables
    """Simulates the circuit, counting how many measurements were determined.

    In most cases, for a quantum error correcting code, the result should be
    related to the number of detectors plus the number of observables declared
    in the circuit.
    """

    num_determined_measurements = 0
    sim = stim.TableauSimulator()
    n = circuit.num_qubits

    def run_block(block: stim.Circuit, reps: int):
        nonlocal num_determined_measurements
        for _ in range(reps):
            for inst in block:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    run_block(inst.body_copy(), inst.repeat_count)
                elif inst.name == 'M' or inst.name == 'MR':
                    args = inst.gate_args_copy()
                    for t in inst.targets_copy():
                        assert t.is_qubit_target
                        known = sim.peek_z(t.value) != 0
                        num_determined_measurements += known
                        sim.do(stim.CircuitInstruction(inst.name, [t.value], args))
                elif inst.name == 'MX' or inst.name == 'MRX':
                    args = inst.gate_args_copy()
                    for t in inst.targets_copy():
                        assert t.is_qubit_target
                        known = sim.peek_x(t.value) != 0
                        num_determined_measurements += known
                        sim.do(stim.CircuitInstruction(inst.name, [t.value], args))
                elif inst.name == 'MY' or inst.name == 'MRY':
                    args = inst.gate_args_copy()
                    for t in inst.targets_copy():
                        assert t.is_qubit_target
                        known = sim.peek_y(t.value) != 0
                        num_determined_measurements += known
                        sim.do(stim.CircuitInstruction(inst.name, [t.value], args))
                elif inst.name == 'MPP':
                    args = inst.gate_args_copy()
                    targets = inst.targets_copy()
                    start = 0
                    while start < len(targets):
                        end = start + 1
                        while end < len(targets) and targets[end].is_combiner:
                            end += 2

                        p = stim.PauliString(n)
                        for t in targets[start:end:2]:
                            if t.is_x_target:
                                p[t.value] = 'X'
                            elif t.is_y_target:
                                p[t.value] = 'Y'
                            elif t.is_z_target:
                                p[t.value] = 'Z'
                            else:
                                raise NotImplementedError(f'{t=} {inst=}')

                        known = sim.peek_observable_expectation(p) != 0
                        num_determined_measurements += known
                        sim.do(stim.CircuitInstruction(inst.name, targets[start:end], args))

                        start = end
                else:
                    sim.do(inst)

    run_block(circuit, 1)
    return num_determined_measurements

len(ex.circ.shortest_graphlike_error())
count_determined_measurements_in_circuit(ex.circ)
