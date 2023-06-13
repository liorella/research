##
import stim

## first stabilizer round for initializing in logical +
circuit = stim.Circuit('''
R 0 1 2 3 4 5 6 7 8
R 9 10 11 12 13 14 15 16
TICK
H 0 1 2 3 4 5 6 7 8 13 14 15 16 
TICK
CX 0 9 13 3 6 11 4 10 16 7 14 5 
TICK
CX 1 9 13 0 7 11 5 10 16 4 14 2 
TICK
CX 15 6 13 4 3 11 1 10 16 8 7 12
TICK
CX 15 3 13 1 4 11 2 10 16 5 8 12
TICK
H 13 14 15 16 
''')
# Create a stabilizer tableau

tableau = stim.Tableau.from_circuit(circuit, ignore_measurement=True, ignore_reset=True)
print(repr(tableau))

## taking all stabilizers and converting into a circuit - first stabilizer is added manually
tableau2 = stim.Tableau.from_stabilizers([
    stim.PauliString("+XXX______"),
    stim.PauliString("+ZZ_______"),
    stim.PauliString("+_ZZ_ZZ___"),
    stim.PauliString("+___ZZ_ZZ_"),
    stim.PauliString("+_______ZZ"),
    stim.PauliString("+XX_XX____"),
    stim.PauliString("+__X__X___"),
    stim.PauliString("+___X__X__"),
    stim.PauliString("+____XX_XX"),
])

circuit2 = tableau2.to_circuit(method="elimination")

## adding to the circuit another stabilizer round to retrieve similar line as before
circuit3 = stim.Circuit('''
    H 0
    CX 0 1
    H 5
    CX 5 0 1 2 5 1
    H 6
    CX 6 2
    H 7
    CX 4 3 5 3 7 3 5 4 4 5 5 4
    H 8
    CX 8 4 6 5 5 6 6 5 8 5 7 6 6 7 7 6 8 7

    H 13 14 15 16 
    TICK
    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    TICK
    CX 1 9 13 0 7 11 5 10 16 4 14 2 
    TICK
    CX 15 6 13 4 3 11 1 10 16 8 7 12
    TICK
    CX 15 3 13 1 4 11 2 10 16 5 8 12
    TICK
    H 13 14 15 16 

''')
tableau3 = stim.Tableau.from_circuit(circuit3)
print(repr(tableau3))

## two independent surfaces
circuit4 = stim.Circuit('''
    H 0
    CX 0 1
    H 5
    CX 5 0 1 2 5 1
    H 6
    CX 6 2
    H 7
    CX 4 3 5 3 7 3 5 4 4 5 5 4
    H 8
    CX 8 4 6 5 5 6 6 5 8 5 7 6 6 7 7 6 8 7

    H 24
    CX 24 25
    H 29
    CX 29 24 25 26 29 25
    H 30
    CX 30 26
    H 31
    CX 28 27 29 27 31 27 29 28 28 29 29 28
    H 32
    CX 32 28 30 29 29 30 30 29 32 29 31 30 30 31 31 30 32 31


''')
tableau4 = stim.Tableau.from_circuit(circuit4)
print(repr(tableau4))

## first round of surgery
circuit5 = stim.Circuit('''
    H 0
    CX 0 1
    H 5
    CX 5 0 1 2 5 1
    H 6
    CX 6 2
    H 7
    CX 4 3 5 3 7 3 5 4 4 5 5 4
    H 8
    CX 8 4 6 5 5 6 6 5 8 5 7 6 6 7 7 6 8 7

    H 24
    CX 24 25
    H 29
    CX 29 24 25 26 29 25
    H 30
    CX 30 26
    H 31
    CX 28 27 29 27 31 27 29 28 28 29 29 28
    H 32
    CX 32 28 30 29 29 30 30 29 32 29 31 30 30 31 31 30 32 31

    H 13 14 15 16 20 21 22 23 37 38 39 40  

    CX 0 9 13 3 6 11 4 10 16 7 14 5 
    CX 24 33 37 27 30 35 28 34 40 31 38 29 
    CX 18 12 21 19 20 17 23 25

    CX 1 9 13 0 7 11 5 10 16 4 14 2
    CX 25 33 29 34 31 35 37 24 38 26 40 28 
    CX 23 18 21 8 20 6 19 12  

    CX 15 6 13 4 3 11 1 10 16 8 7 12
    CX 39 30 37 28 40 32 25 34 27 35 31 36
    CX 20 18 22 24 17 33 23 26

    CX 15 3 13 1 4 11 2 10 16 5 8 12
    CX 39 27 28 35 40 29 32 36 26 34 37 25
    CX 20 7 22 17 18 33 23 19

    H 13 14 15 16 20 21 22 23 37 38 39 40


''')
tableau5 = stim.Tableau.from_circuit(circuit5)
print(repr(tableau5))
## taking all stabilizers after one surgery round into a circuit
tableau6 = stim.Tableau.from_stabilizers([
    stim.PauliString("+XXX__________________"),
    stim.PauliString("+ZZ___________________"),
    stim.PauliString("+_ZZ_ZZ_______________"),
    stim.PauliString("+___ZZ_ZZ_____________"),
    stim.PauliString("+_______ZZ_ZZ_________"),
    stim.PauliString("+XX_XX________________"),
    stim.PauliString("+__X__X_______________"),
    stim.PauliString("+___X__X______________"),
    stim.PauliString("+____XX_XX____________"),
    stim.PauliString("+____________XXX______"),
    stim.PauliString("+_________ZZ_ZZ_______"),
    stim.PauliString("+_____________ZZ_ZZ___"),
    stim.PauliString("+_______________ZZ_ZZ_"),
    stim.PauliString("+___________________ZZ"),
    stim.PauliString("+____________XX_XX____"),
    stim.PauliString("+______________X__X___"),
    stim.PauliString("+_______________X__X__"),
    stim.PauliString("+________________XX_XX"),
    stim.PauliString("+_________XXX_________"),
    stim.PauliString("+__________XX_XX______"),
    stim.PauliString("+______XX_XX__________"),
])

circuit6 = tableau6.to_circuit(method="elimination")

## checking what happens with the surgery data qubits
circuit7 = stim.Circuit('''
    H 0
    CX 0 1
    H 5
    CX 5 0 1 2 5 1
    H 6
    CX 6 2
    H 7
    CX 4 3 5 3 7 3 10 3 11 3 5 4 4 5 5 4
    H 8
    CX 8 4 6 5 5 6 6 5 8 5 7 6 6 7 7 6
    H 20
    CX 20 6 8 7 10 7 11 7 20 7 20 9 9 20 20 9 9 10
    H 18
    CX 18 9
    H 19
    CX 11 10 18 10 19 10 18 11 11 18 18 11 19 11 20 12 12 20 20 12
    H 12
    CX 12 19
    H 14
    CX 14 12 18 13 13 18 18 13 14 13 19 13 19 14 14 19 19 14
    H 15
    CX 15 14 20 15 15 20 20 15
    H 16
    CX 16 15 18 15 19 15 19 16 16 19 19 16
    H 17
    CX 17 16 20 17 17 20 20 17 20 17 19 18 18 19 19 18 20 19
    H 9 10 11
''')
tableau7 = stim.Tableau.from_circuit(circuit7)
print(repr(tableau7))

##
tableau = stim.Tableau(3)
cnot = stim.Tableau.from_named_gate("CNOT")
H = stim.Tableau.from_named_gate("H")
circuit2 = tableau.to_circuit(method="elimination")