round=0 #defines which stabilizer round
# initialize surface1
for data_qubit in range(surface1.data_qubits): # meta-programming (parallel)
    restart(data_qubit)
# perform d stabilizer rounds for surfaces
for i in range(surface1.distance):
    for ancilla_qubit in range(surface1.ancilla_qubits + surface2.ancilla_qubits):# meta-programming (parallel)
        stabilizer_round(ancilla_qubit, readout_element[ancilla_qubit])
        ID=[ancilla_qubit, round]
        send_to_decoder(readout_element[ancilla_qubit], ID)
    round+=1

# perform lattice surgery
for data_qubit in range(surgery12.data_qubits): # meta-programming (parallel)
    restart(data_qubit)
for i in range(surface1.distance):
    for ancilla_qubit in range(
            surface1.ancilla_qubits + surface2.ancilla_qubits + surgery12.ancilla_qubits):# meta-programming (parallel)
        stabilizer_round(ancilla_qubit, readout_element[ancilla_qubit])
        ID=[ancilla_qubit, round]
        send_to_decoder(readout_element[ancilla_qubit], ID)
    round+=1
for surgery_data_qubit in range(surgery12.data_qubits): # meta-programming (parallel)
    measure(surgery_data_qubit,readout_element[surgery_data_qubit])
    ID = [surgery_data_qubit,round-1]
    send_to_decoder(readout_element[surgery_data_qubit], ID, timestamp->re_time[surgery_data_qubit])

# get logical values of lattice surgery
[surgery_result, accuracy] = get_decoding_result(surgery12)

# perform stabilizer rounds until confident in logical results
while accuracy<accuracy_threshold:
    for ancilla_qubit in range(surface1.ancilla_qubits + surface2.ancilla_qubits): # meta-programming (parallel)
        stabilizer_round(ancilla_qubit, readout_element[ancilla_qubit])
        ID=[ancilla_qubit, round]
        send_to_decoder(readout_element[ancilla_qubit], ID)
    [surgery_result, accuracy] = get_decoding_result(surgery12)
    round+=1

#apply a logical pi pulse on surface2 if the logical surgery result is 1
if surgery_result:
    for data_qubit in range(surface2.data_qubits[0:surface2.distance]):# meta-programming (parallel)
        play(data_qubit,pi_pulse,timestamp->ce_time[data_qubit])

# perform stabilizer rounds until at least d rounds after surgery
while round<3*surface1.distance:
    for ancilla_qubit in range(surface1.ancilla_qubits + surface2.ancilla_qubits):# meta-programming (parallel)
        stabilizer_round(ancilla_qubit, readout_element[ancilla_qubit])
        ID = [ancilla_qubit, round]
        send_to_decoder(readout_element[ancilla_qubit], ID)
    round += 1

# measure data qubits of surface 1 and 2
for surface_data_qubit in range(surface1.data_qubits + surface2.data_qubits):# meta-programming (parallel)
    measure(surface_data_qubit,readout_element[surface_data_qubit])
    ID = [surface_data_qubit,round-1]
    send_to_decoder(readout_element[surface_data_qubit], ID)

# get the final logical results
[[logical_surface1,logical_surface2], accuracy] = get_decoding_result(surface1,surface2)

if any(accuracy<accuracy_threshold):
    discard_experiment

#checking online decoder with post-processing
[surgery_check,surface1_check,surface2_check]=post_processing_verification([surgery12,logical_surface1,logical_surface2])
if [surgery_check,surface1_check,surface2_check] !=  [surgery_result,logical_surface1,logical_surface2]:
    discard_experiment


