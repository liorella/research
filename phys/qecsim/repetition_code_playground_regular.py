## The simplest implementation of a repetition code using stim.
import numpy as np

import stim
import pymatching
import matplotlib.pyplot as plt
import networkx as nx
##
rounds=2
distance=5
p=0.1
circuit = stim.Circuit.generated(
    "repetition_code:memory",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p
    )
## extracting the detector-error model
model = circuit.detector_error_model(decompose_errors=True)

## Building the matching graph
matching = pymatching.Matching.from_detector_error_model(model)


## plotting the matching graph

plot=False
if plot:
    E=matching.edges() # edges and wieghts
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


## simulation of the repetition code
shots=1000
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

num_errors = 0
for i in range(shots):
    [predicted_observables, weight] = matching.decode(syndrome[i, :], return_weight=True)
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)

log_error_prob=num_errors/shots
print(log_error_prob)