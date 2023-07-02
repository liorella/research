import fusion_blossom as fb

code = fb.CodeCapacityPlanarCode(d=3, p=0.4, max_half_weight=500)

syndrome = code.generate_random_errors(seed=0)
print(syndrome)

##
visualizer = None
if True:  # change to False to disable visualizer for faster decoding
    visualize_filename = fb.static_visualize_data_filename()
    positions = code.get_positions()
    visualizer = fb.Visualizer(filepath=visualize_filename, positions=positions)

initializer = code.get_initializer()
solver = fb.SolverSerial(initializer)
