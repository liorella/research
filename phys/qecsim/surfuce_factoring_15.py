import abc
import dataclasses
from enum import Enum

import stim
import pymatching
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List


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


class Surface:
    def __init__(self, dist: int):
        self.dist = dist
        self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        self.data_qubits = np.zeros((dist, dist), dtype=int)
        self.ancilla_qubits = {'L': np.zeros(((dist-1), ), dtype=int),
                               'R': np.zeros(((dist-1), ), dtype=int),
                               'T': np.zeros(((dist-1), ), dtype=int),
                               'B': np.zeros(((dist-1), ), dtype=int),
                               'C': np.zeros((dist-1, dist-1), dtype=int)}

    def flip_orientation(self):
        if self.orientation.value:
            self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        else:
            self.orientation = SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL

    def _allocate_ancillas(self, circ, coord, direction: str, ancilla_name):
        direction_to_number = {'L': 10000, 'T': 10001, 'R': 10002, 'B': 10003}
        for i in range(self.dist-1):
            if ((direction == 'L' or direction == 'T') and (coord[0] + coord[1] + i) % 2 == 0) or \
               ((direction == 'B' or direction == 'R') and (coord[0] + coord[1] + i) % 2 != 0):
                # circ.append('QUBIT_COORDS', [ancilla_name], (coord[0], coord[1], direction_to_number[direction], i))
                self.ancilla_qubits[direction][i] = ancilla_name
                ancilla_name += 1
            else:
                self.ancilla_qubits[direction][i] = -1
        return ancilla_name


    def allocate_qubits(self, circ, coord):
        ancilla_name = coord[0] * 10000 + coord[1] * 1000 + self.dist ** 2
        data_name = coord[0] * 10000 + coord[1] * 1000
        for i in range(self.dist):
            for j in range(self.dist):
                # circ.append('QUBIT_COORDS', [data_name], (coord[0], coord[1], i, j))
                self.data_qubits[i, j] = data_name
                data_name += 1

        for direction in ['L', 'R', 'T', 'B']:
            ancilla_name = self._allocate_ancillas(circ, coord, direction, ancilla_name)

        for i in range(self.dist-1):
            for j in range(self.dist-1):
                # circ.append('QUBIT_COORDS', [ancilla_name], (coord[0], coord[1], 1004, i, j))
                self.ancilla_qubits['C'][i, j] = ancilla_name
                ancilla_name += 1


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
            ancilla_index = (self.dist-1, ancilla_index)
        elif ancilla_direction == 'T':
            ancilla_index = (ancilla_index, self.dist-1)
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
        return None if ret[0] < 0 or ret[1] < 0 or ret[0] >= self.dist or ret[1] >= self.dist else ret

    def _group_ancillas(self):
        even_coord = 1 if self.ancilla_qubits['L'][0] == -1 else 0
        ancilla_in_group0=[]
        ancilla_in_group1=[]
        for i, direction in enumerate(['L', 'R', 'T', 'B']):
            for ancilla in self.ancilla_qubits[direction]:
                if ancilla >= 0:
                    if i<2:
                        ancilla_in_group0.append(ancilla)
                    else:
                        ancilla_in_group1.append(ancilla)
        for i in range(self.dist - 1):
            for j in range(self.dist - 1):
                if (i + j + even_coord) % 2 :
                    ancilla_in_group0.append(self.ancilla_qubits['C'][i, j])
                else:
                    ancilla_in_group1.append(self.ancilla_qubits['C'][i, j])
        return [ancilla_in_group0,ancilla_in_group1]


    def _get_ancilla_with_targets(self, target_direction, stabilizer_group: int): # gets direction of 2 qubit gate and which stabilizer_group (orientation independent), creates pair (source and target qubits)
        ops = []
        directions = ['L', 'R'] if stabilizer_group == 0 else ['T', 'B']
        for direction in directions:
            for i, ancilla in enumerate(self.ancilla_qubits[direction]):
                if ancilla >= 0:
                    target = self._get_target(direction, i, target_direction)
                    if target is not None:
                        ops += ancilla, self.data_qubits[target]

        even_coord = 1 if self.ancilla_qubits['L'][0] == -1 else 0
        for i in range(self.dist-1):
            for j in range(self.dist-1):
                if (i + j + even_coord) % 2 == stabilizer_group:
                    continue
                target = self._get_target('C', (i, j), target_direction)
                if target is not None:
                    ops += self.ancilla_qubits['C'][i, j], self.data_qubits[target]
        return ops


    def _two_qubit_epoch(self, circ, direction, error_model: BaseErrorModel):
        op0 = self._get_ancilla_with_targets(direction, 0)
        op1 = self._get_ancilla_with_targets(direction, 1)
        if self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL:
            circ.append("CX", list(reversed(op1)))
            circ.append("CX", op0)
        else:
            circ.append("CX", op1)
            circ.append("CX", list(reversed(op0)))
        error_model.generate_two_qubit_error(circ, op0 + op1)



    def stabilizer_round(self, circ, epoch: int, measurements: list, error_model: BaseErrorModel):
        ancillas = self._group_ancillas()
        target_for_H = ancillas[0] if self.orientation == SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL else ancillas[1]
        if epoch == 0:
            circ.append("R", ancillas[0]+ancillas[1])
        elif epoch == 1:
            circ.append("H", target_for_H)
            error_model.generate_single_qubit_error(circ, target_for_H)
        elif epoch == 2:
            self._two_qubit_epoch(circ, 'NW', error_model)
        elif epoch == 3:
            self._two_qubit_epoch(circ, 'NE', error_model)
        elif epoch == 4:
            self._two_qubit_epoch(circ, 'SE', error_model)
        elif epoch == 5:
            self._two_qubit_epoch(circ, 'SW', error_model)
        elif epoch == 6:
            circ.append("H", target_for_H)
            error_model.generate_single_qubit_error(circ, target_for_H)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas[0]+ancillas[1])
            circ.append("M", ancillas[0]+ancillas[1])
            measurements.extend(ancillas[0]+ancillas[1])
            for ancilla in (ancillas[0]+ancillas[1]):
                occ = np.where(np.array(measurements) == ancilla)[0]-len(measurements)
                if len(occ) >= 2:
                    circ.append("DETECTOR", [stim.target_rec(occ[-1]), stim.target_rec(occ[-2])])
                else:
                    circ.append("DETECTOR", [stim.target_rec(occ[-1])])




    def _add_measurement_detectors(self, circ: stim.Circuit, basis: MeasurementBasis, measurements: list):
        stabilizer_group = 0 if self.orientation.value == basis.value else 1
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


    def initialize(self, circ, state: initialState, error_model: BaseErrorModel, measurements):
        data_qubits=self.data_qubits.flatten()
        circ.append("R", data_qubits)
        error_model.generate_single_qubit_error(circ, data_qubits)
        if state == initialState.Z_MINUS:
            circ.append("TICK")
            circ.append("X", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
        elif state == initialState.X_PLUS:
            circ.append("TICK")
            circ.append("H", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
        elif state == initialState.X_MINUS:
            circ.append("TICK")
            circ.append("X", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
            circ.append("TICK")
            circ.append("H", data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
        for epoch in range(7):
            self.stabilizer_round(circ, epoch, measurements, error_model)
            circ.append("TICK")
        error_model.generate_measurement_qubit_error(circ, self._all_ancillas())
        circ.append("M", self._all_ancillas())
        measurements.extend(self._all_ancillas())
        circ.append("TICK")
        if state == initialState.Z_PLUS or state == initialState.Z_MINUS:
            ancillas = self._group_ancillas()[self.orientation.value]
        elif state == initialState.X_PLUS or state == initialState.X_MINUS:
            ancillas = self._group_ancillas()[not(self.orientation.value)]
        for ancilla in ancillas:
            occ = np.where(np.array(measurements) == ancilla)[0] - len(measurements)
            circ.append("DETECTOR", [stim.target_rec(occ)])

    def print_surface_name(self):
        print(self.data_qubits)



class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface], error_model: BaseErrorModel):
        self.circ = stim.Circuit()
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(self.circ, coordinate)
        self.surfaces = surfaces
        self.measurements = []
        self.error_model = error_model

    def __getitem__(self, coor):
        return self.surfaces[coor]

    def flip_surface_orientation(self, coor: tuple):
        self.surfaces[coor].flip_orientation()

    def measure_surface(self, coor: tuple, basis: MeasurementBasis, observable_index):
        self.surfaces[coor].measurement(self.circ, basis, self.error_model, self.measurements, observable_index)
    def Initialize_surface(self, coor: tuple, state: initialState):
        self.surfaces[coor].initialize(self.circ, state, self.error_model, self.measurements)

    def stabilizer_round(self):
        for epoch in range(8):
            for surface in self.surfaces.values():
                surface.stabilizer_round(self.circ, epoch, self.measurements, self.error_model)
            self.circ.append("TICK")


##
error_model = ErrorModel(single_qubit_error=0.001, two_qubit_error=0.03, measurement_error=0.05)
# error_model = NoErrorModel()
d=5
ex = Experiment({
    (0,0): Surface(d),
     (1,0): Surface(d),
    # (0,1): Surface(d)
}, error_model)


##
#ex[0, 0].flip_orientation()

ex.Initialize_surface((0,0), initialState.Z_MINUS)
ex.Initialize_surface((1,0), initialState.Z_MINUS)
ex.stabilizer_round()
# ex.stabilizer_round()
ex.measure_surface((0, 0), MeasurementBasis.Z_BASIS, observable_index=0)
ex.measure_surface((1, 0), MeasurementBasis.Z_BASIS, 1)

print(ex.circ)
# ex.circ.diagram()
##
model = ex.circ.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

sampler = ex.circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)

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
