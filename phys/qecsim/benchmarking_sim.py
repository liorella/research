import matplotlib.pyplot as plt
import numpy as np

# Given parameters
L0 = 10
delta_values = [1, 10]  # Values of delta0
g_values = [0.9, 1.1, 1.3]  # Values of g

M_anc_values = [0, 1]  # Values of delta0

# Range of n values
n_values = np.arange(1, 41)  # Updated range of n values

# Define distinct colors for similar g values
colors = [['#66c2a5', '#fc8d62', '#8da0cb'],['#4c8c7e', '#e15c39', '#6e84a8']]

markers = ['o', 's', '^', 'd']
delta_label = r'$\delta_{0}$'

# Create a plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for k, M_anc in enumerate(M_anc_values):
    for i, delta0 in enumerate(delta_values):
        for j, g in enumerate(g_values):
            # Calculate function values for the given delta0 and g
            function_values = L0 + ((delta0 - M_anc) * g / (g - 1)) * (g ** n_values - 1)
            ratio_values = function_values[1:] / function_values[:-1]
            # Plot the data with distinct markers and colors
            if k == 0:
                label = f'{delta_label}={delta0}, g={g:.1f}, $M_{{anc}}<8$'
                axs[0].plot(n_values, function_values, marker=markers[i], linestyle='-', color=colors[k][j], label=label)
                axs[1].plot(n_values[1:], ratio_values, marker=markers[i], linestyle='-', color=colors[k][j],
                            label=label)
            else:
                label = f'{delta_label}={delta0}, g={g:.1f}, $M_{{anc}}=8$'
                axs[0].plot(n_values, function_values, marker=markers[i], linestyle='-', color=colors[k][j], label=label)
                axs[1].plot(n_values[1:], ratio_values, marker=markers[i], linestyle='-', color=colors[k][j],
                            label=label)

# Annotate the color coding for g
axs[0].text(0, 10**5.2, 'Green: g=0.9', color=colors[0][0], fontsize=12)
axs[0].text(0, 10**5.5, 'Orange: g=1.1', color=colors[0][1], fontsize=12)
axs[0].text(0, 10**5.8, 'Blue: g=1.3', color=colors[0][2], fontsize=12)

# Annotate the color coding for delta0
axs[0].text(0, 10**4.8, 'Light Color: $M_{anc}<8$', fontsize=12)
axs[0].text(0, 10**4.5, 'Dark Color: $M_{anc}=8$', fontsize=12)

axs[0].text(0, 10**4, 'Squares: $\delta_{0}=10$', fontsize=12)
axs[0].text(0, 10**3.7, 'Circles: $\delta_{0}=1$', fontsize=12)

axs[0].set_xlabel('round', fontsize=14)
axs[0].set_ylabel(r'Feedback latency [$T_{s}$]', fontsize=14)
axs[0].set_yscale('log')  # Set y-axis to log scale
axs[0].grid(True)
axs[1].grid(True)
axs[1].set_xlabel('round', fontsize=14)
axs[1].set_ylabel(r'FLAG', fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size
axs[0].tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size


# Given parameters
tau0 = 8
r = 7
Lth = 2 * r

# Range of g values (smaller than 1)
g = np.linspace(0, 1.3, 200)  # Adjust the range as needed

L0 = tau0 + 2 * g * r
delta0 = L0 - Lth

# Function to calculate the function values for each g
function_values = L0 + delta0 * g / (1 - g) * np.heaviside(delta0, 1)
circuit_delay = delta0 / (1 - g) * np.heaviside(delta0, 1)

# Create the plot

# Plot function values
axs[2].plot(g, function_values, label='$L_{ss}$')

# Plot circuit delay on a secondary y-axis
axs[2] = plt.gca()
ax2 = axs[2].twinx()
ax2.plot(g, circuit_delay, color='tab:orange', linestyle='dashed', label='Circuit Delay')
ax2.set_ylabel('Circuit Delay $[T_{s}]$', color='tab:orange', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Set common x-axis label
axs[2].set_xlabel('g', fontsize=14)

# Set y-axis labels and limits
axs[2].set_ylabel('$L_{ss}  [T_{s}]$', fontsize=14)
axs[2].set_ylim(0, 100)
ax2.set_ylim(0, 100)

# Add a horizontal line at Lth
axs[2].axhline(y=Lth, color='gray', linestyle='--')
axs[2].text(0.2, Lth + 2, '$L_{th}$', color='gray', fontsize=12)

# Shade the areas where conditions are met
axs[2].axvspan(1, 1.3, facecolor='pink', alpha=0.2)
axs[2].axvspan(0, (Lth - tau0) / (2 * r), facecolor='blue', alpha=0.2)

axs[2].text(1.05, 40, '$L->\infty$', color='red', fontsize=12)
axs[2].text(0.1, 40, '$L<L_{th}$', color='Blue', fontsize=12)

# Set x-axis limits
axs[2].set_xlim(0, 1.3)

# Combine legends from both y-axes
lines, labels = axs[2].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Add gridlines
axs[2].grid(True)

axs[2].tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size

# Show the plot
plt.show()