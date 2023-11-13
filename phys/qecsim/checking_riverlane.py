import numpy as np
import matplotlib.pyplot as plt

defect_per_round = np.array([0.0135, 0.025, 0.04, 0.051, 0.063])
d = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])




# Your decoding_time array remains the same
decoding_time = np.array([
     [3 *0.078, 3 *0.08, 3 *0.082, 3 *0.085, 3 *0.09],
     [5 *0.06, 5 *0.07, 5 *0.08, 5 *0.095, 5 *0.12],
     [7 *0.06, 7 *0.087, 7 *0.13, 7 *0.17, 7 *0.22],
     [9 * 0.075, 9 * 0.15,9 *  0.23, 9 * 0.33,9 *  0.43],
     [11 *0.11, 11 *0.25, 11 *0.4, 11 *0.6, 11 *0.85],
     [13 *0.18, 13 *0.4, 13 *0.75,13 *1.1, 13 *1.8],
     [15 * 0.26, 15 * 0.68, 15 * 1.3, 15 * 2,15 *  3],
     [17 *0.38, 17 *1.05, 17 *2.2, 17 *3.5, 17 *5.5],
     [19 * 0.65,19 *  1.8, 19 * 3.5, 19 * 5.9,19 * 9.5],
     [21 *0.85, 21 *2.8, 21 *5.5,21 * 9.5, 21 *15],
     [23 *1.3,23 * 4, 23 *8.5,23 * 15, 23 *25]])

# Calculate the matrix
total_vertices = np.outer(d * ((d - 1) / 2) ** 2, defect_per_round)

# Print the resulting matrix
print(total_vertices)


# Define a list of unique colors for each row
colors = plt.cm.viridis(np.linspace(0, 1, len(d)))  # You can choose any colormap

# Create the plot
plt.figure(figsize=(10, 6))

for i in range(len(d)):
    plt.scatter(total_vertices[i], decoding_time[i], marker='o', c=[colors[i]], label=f'd = {d[i]}')

plt.xlabel('Total Vertices')
plt.ylabel('Decoding Time')
plt.title('Decoding Time vs. Total Vertices')
plt.grid(True)
plt.legend()
plt.yscale('log')  # Set y-axis to a logarithmic scale
plt.xscale('log')  # Set y-axis to a logarithmic scale

plt.show()

syndromes=total_vertices.flatten()
time=decoding_time.flatten()
tau0=0.25
tau1=0.13
alpha=1.58

latency= tau0+tau1*syndromes**alpha

plt.plot(syndromes,latency)