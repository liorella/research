
import fusion_blossom as fb
import numpy as np
import stim
import pymatching
d=5
p=0.01
num_shots = 10
## defining simulations
num_shots = 50000
d_vec =[3,5,7,9]
p_vec=[0.0025,0.005,0.0075,0.01,0.025,0.05]
# p_vec=[0.005,0.01,0.025]
fb_errors_mat=np.zeros((len(d_vec),len(p_vec)))
pymatching_errors_mat=np.zeros((len(d_vec),len(p_vec)))

for d_ind,d in enumerate(d_vec):

    def from_subgraph_to_logical_flip(subgraph, d):
        edges_in_unitcell=int(d ** 2+(d+1)**2/2)
        flips_in_unitcell = np.zeros((edges_in_unitcell, 1))
        for edge_index in subgraph:
            flips_in_unitcell[edge_index % edges_in_unitcell] += 1
        dataqubit_flips = flips_in_unitcell[0:d ** 2] % 2
        return bool(sum(dataqubit_flips[0::d]) % 2)

    for p_ind,p in enumerate(p_vec):
        #
        rounds = d

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=d,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p)

        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)

        sampler = circuit.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

        predicted_pymatching_observables = matching.decode_batch(syndrome)
        pymatching_errors = sum(predicted_pymatching_observables != actual_observables)

        #defining masks and indices to map from stim syndrome list to fusion blossom list
        syndrome_int=syndrome[0].astype(int)
        mask=np.ones((1, len(syndrome_int)), dtype=bool)
        false_vec=np.zeros((1, int(d/2-1/2)), dtype=bool)
        for i in range(rounds-1):
            for j in range(d+1):
                if (j%d)==0:
                    mask[0][int((d**2-1)/2+(d**2-1)*i+(j//d)*(d**2-(d-1)/2-1)):int((d**2-1)/2+(d**2-1)*i+(j//d)*(d**2-(d-1)/2-1)+(d-1)/2)]=false_vec
                else:
                    mask[0][int((d**2-1)/2+(d**2-1)*i+(d-1)/2+1+(j-1)*d):int((d**2-1)/2+(d**2-1)*i+(d-1)/2+1+(j-1)*d+d-1):2]=false_vec

#
        fusion_blossom_vertex_list=np.zeros(int((d**2-1)/2)*(rounds+1))
        for i in range(len(fusion_blossom_vertex_list)):
            stab_round = i // ((d**2-1)/2)
            if stab_round%rounds==0:
                row=d-(i-stab_round*((d**2-1)/2))//((d-1)/2)
                col=(i-stab_round*((d**2-1)/2))%((d-1)/2) #in stim
                fusion_blossom_vertex_list[i]=row*(d+1)/2+col+row%2+stab_round*(d+1)*(d+1)/2
            else:
                row=(d-1)/2-((i-stab_round*((d**2-1)/2))%((d+1)/2))
                col=(i-stab_round*((d**2-1)/2))//((d+1)/2)
                row=2*row+col%2
                fusion_blossom_vertex_list[i]=stab_round*((d+1)**2)/2+row*(d+1)/2+row%2+col//2



        code=fb.PhenomenologicalRotatedCode(d=d, noisy_measurements=rounds, p=p, max_half_weight=500)
        fusion_syndrome = code.generate_random_errors(seed=1000)
        initializer = code.get_initializer()
        # visualizer = None
        #  if True:  # change to False to disable visualizer for faster decoding
        #      visualize_filename = fb.static_visualize_data_filename()
        #      positions = code.get_positions()
        #      visualizer = fb.Visualizer(filepath=visualize_filename, positions=positions)
        #  initializer = code.get_initializer()
        #  solver = fb.SolverSerial(initializer)
        #  solver.solve(fusion_syndrome)
        #  subgraph = solver.subgraph(visualizer)
        #  if visualizer is not None:
        #      fb.print_visualize_link(filename=visualize_filename)
        #      fb.helper.open_visualizer(visualize_filename, open_browser=True)

        predicted_fb_observables=np.zeros((1,len(predicted_pymatching_observables)),dtype=bool)[0]
        fb_errors=0
        for i in range(num_shots):
            syndrome_int_z = syndrome[i].astype(int)[mask[0]]
            fusion_syndrome.defect_vertices=fusion_blossom_vertex_list[syndrome_int_z.astype(bool)].astype(int)
            solver = fb.SolverSerial(initializer)
            solver.solve(fusion_syndrome)
            subgraph = solver.subgraph()

            fb_errors+= from_subgraph_to_logical_flip(subgraph,d) != actual_observables[i]


        # print(f"errors without a decoder = {sum(actual_observables)} from {num_shots} shots")
        # print(f"errors with pymatching = {pymatching_errors}")
        # print(f"errors with fusion blossom = {fb_errors}")
         print(f"d={d},p={p} errors rate of fb is = {fb_errors/num_shots} and of pymatching is {pymatching_errors/num_shots}")

        fb_errors_mat[d_ind,p_ind]=fb_errors/num_shots
        pymatching_errors_mat[d_ind,p_ind]=pymatching_errors/num_shots
##

import matplotlib.pyplot as plt

for i in range(fb_errors_mat.shape[0]):
    plt.plot(p_vec,fb_errors_mat[i], label=f'fb d= {d_vec[i]}',marker='o')
    plt.plot(p_vec,pymatching_errors_mat[i], label=f'pymatching d= {d_vec[i]}',marker='+')


plt.xlabel('physical error')
plt.ylabel('logical error')
plt.legend()
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale

plt.show()

