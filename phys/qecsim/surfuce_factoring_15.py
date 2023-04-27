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
    Z_BASIS = 1
    X_BASIS = 2

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
                circ.append('QUBIT_COORDS', [ancilla_name], (coord[0], coord[1], direction_to_number[direction], i))
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
                circ.append('QUBIT_COORDS', [data_name], (coord[0], coord[1], i, j))
                self.data_qubits[i, j] = data_name
                data_name += 1

        for direction in ['L', 'R', 'T', 'B']:
            ancilla_name = self._allocate_ancillas(circ, coord, direction, ancilla_name)

        for i in range(self.dist-1):
            for j in range(self.dist-1):
                circ.append('QUBIT_COORDS', [ancilla_name], (coord[0], coord[1], 1004, i, j))
                self.ancilla_qubits['C'][i, j] = ancilla_name
                ancilla_name += 1


        # if ((coord[0] + coord[1]) % 2) == 0:
        #     for i in range(self.dist):
        #         for j in range(self.dist):
        #             if i==0 and j<(self.dist-2) and not(j%2):
        #                 circ.append('QUBIT_COORDS', [ancilla_name], (coord[0], coord[1], 1000, j))
        #                 self.ancilla_qubits['L'][j] = ancilla_name
        #                 ancilla_name += 1
        #             elif i % 2:
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j + 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #             elif i<(self.dist-1) and i>0:
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j - 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #             elif i==(self.dist-1):
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j - 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #     for j in range(1, self.dist,2):
        #         circ.append('QUBIT_COORDS', [ancilla_name],
        #                     (shift_surf_x + 2 * self.dist - 1, shift_surf_y + 2 * j + 1))
        #         self.ancilla_qubits.append(ancilla_name)
        #         ancilla_name += 1
        #
        # else:
        #      for i in range(self.dist):
        #          for j in range(self.dist):
        #             if i == 0 and j < (self.dist - 2) and not(j%2):
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j + 3))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #             elif i % 2:
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j - 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #             elif i < (self.dist - 1) and i > 0:
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j + 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #             elif i == (self.dist - 1):
        #                 circ.append('QUBIT_COORDS', [ancilla_name],
        #                             (shift_surf_x + 2 * i - 1, shift_surf_y + 2 * j + 1))
        #                 self.ancilla_qubits.append(ancilla_name)
        #                 ancilla_name += 1
        #      for j in range(1, self.dist,2):
        #          circ.append('QUBIT_COORDS', [ancilla_name],
        #                      (shift_surf_x + 2 * self.dist -1, shift_surf_y + 2 * j - 1))
        #          self.ancilla_qubits.append(ancilla_name)
        #          ancilla_name += 1

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

    def _get_ancilla_with_tragets(self, target_direction, stabilizer_group: int): # gets direction of 2 qubit gate and which stabilizer_group (orientation independent), creates pair (source and target qubits)
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

    def _two_qubit_epoch(self, circ, direction):
        op0 = self._get_ancilla_with_tragets(direction, 0)
        op1 = self._get_ancilla_with_tragets(direction, 1)
        if self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL:
            circ.append("CZ", op0)
            circ.append("CX", op1)
        else:
            circ.append("CZ", op1)
            circ.append("CX", op0)

    def stabilizer_round(self, circ, epoch: int):
        if epoch == 0:
            circ.append("R", self._all_ancillas())
        elif epoch == 1:
            circ.append("H", self._all_ancillas())
        elif epoch == 2:
            self._two_qubit_epoch(circ, 'NW')
        elif epoch == 3:
            self._two_qubit_epoch(circ, 'NE')
        elif epoch == 4:
            self._two_qubit_epoch(circ, 'SE')
        elif epoch == 5:
            self._two_qubit_epoch(circ, 'SW')
        elif epoch == 6:
            circ.append("H", self._all_ancillas())
        elif epoch == 7:
            circ.append("M", self._all_ancillas())

    def initialize(self, circ, state):
        pass

    def measurement(self, circ: stim.Circuit, basis: MeasurementBasis):
        if basis == MeasurementBasis.Z_BASIS:
            circ.append('MZ',self.data_qubits.flatten())
        elif basis == MeasurementBasis.X_BASIS:
            circ.append('H', self.data_qubits.flatten())
            circ.append("Tick")
            circ.append('MZ', self.data_qubits.flatten())
        #circ.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4), stim.target_rec(-6)])
        #circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(- 1)], (0))

    def print_surface_name(self):
        print(self.data_qubits)



class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface]):
        self.circ = stim.Circuit()
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(self.circ, coordinate)

        self.surfaces = surfaces

    def __getitem__(self, coor):
        return self.surfaces[coor]

    def flip_surface_orientation(self, coor: tuple):
        self.surfaces[coor].flip_orientation()

    def measure_surface(self, coor: tuple, basis: MeasurementBasis):
        self.surfaces[coor].measurement(self.circ, basis)

    def stabilizer_round(self):
        for epoch in range(8):
            for surface in self.surfaces.values():
                surface.stabilizer_round(self.circ, epoch)
            self.circ.append("TICK")



##
d=3
ex = Experiment({
    (0,0): Surface(d),
    (1,0): Surface(d)
})


##

ex[0, 0].flip_orientation()
ex.stabilizer_round()
print(ex.circ)

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
