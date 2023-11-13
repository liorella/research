import math

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
fig0, axs0 = plt.subplots()
fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots()
for k, M_anc in enumerate(M_anc_values):
    for i, delta0 in enumerate(delta_values):
        for j, g in enumerate(g_values):
            # Calculate function values for the given delta0 and g
            function_values = L0 + ((delta0 - M_anc) * g / (g - 1)) * (g ** n_values - 1)
            ratio_values = function_values[1:] / function_values[:-1]
            # Plot the data with distinct markers and colors
            if k == 0:
                label = f'{delta_label}={delta0}, g={g:.1f}, $M_{{anc}}<8$'
                axs0.plot(n_values, function_values, marker=markers[i], linestyle='-', color=colors[k][j], label=label)
                axs2.plot(n_values[1:], ratio_values, marker=markers[i], linestyle='-', color=colors[k][j],
                            label=label)
            else:
                label = f'{delta_label}={delta0}, g={g:.1f}, $M_{{anc}}=8$'
                axs0.plot(n_values, function_values, marker=markers[i], linestyle='-', color=colors[k][j], label=label)
                axs2.plot(n_values[1:], ratio_values, marker=markers[i], linestyle='-', color=colors[k][j],
                            label=label)

# Annotate the color coding for g
axs0.text(0, 10**5.2, 'Green: g=0.9', color=colors[0][0], fontsize=12)
axs0.text(0, 10**5.5, 'Orange: g=1.1', color=colors[0][1], fontsize=12)
axs0.text(0, 10**5.8, 'Blue: g=1.3', color=colors[0][2], fontsize=12)

# Annotate the color coding for delta0
axs0.text(0, 10**4.8, 'Light Color: $M_{anc}<8$', fontsize=12)
axs0.text(0, 10**4.5, 'Dark Color: $M_{anc}=8$', fontsize=12)

axs0.text(0, 10**4, 'Squares: $L_{0}=20$', fontsize=12)
axs0.text(0, 10**3.7, 'Circles: $L_{0}=11$', fontsize=12)
axs0.text(0, 10**3.5, 'r=10', fontsize=12)

axs0.set_xlabel('gate number', fontsize=14)
axs0.set_ylabel(r'feedforward latency [$T_{s}$]', fontsize=14)
axs0.set_yscale('log')  # Set y-axis to log scale
axs0.grid(True)
axs2.grid(True)
axs2.set_xlabel('gate number', fontsize=14)
axs2.set_ylabel(r'feedforward latency ratio', fontsize=14)
axs2.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size
axs0.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size


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
axs1.plot(g, function_values, label='$L_{ss}$')

# Plot circuit delay on a secondary y-axis
# axs1 = plt.gca()
ax2 = axs1.twinx()
ax2.plot(g, circuit_delay, color='tab:orange', linestyle='dashed', label='Circuit Delay')
ax2.set_ylabel('Circuit Delay per gate $[T_{s}]$', color='tab:orange', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Set common x-axis label
axs1.set_xlabel('decoding throughput (g)', fontsize=14)

# Set y-axis labels and limits
axs1.set_ylabel('Steady state feedforward latency $[T_{s}]$', fontsize=14)
axs1.set_ylim(0, 100)
ax2.set_ylim(0, 100)

# Add a horizontal line at Lth
axs1.axhline(y=Lth, color='gray', linestyle='--')
axs1.text(0.2, Lth + 2, '$L_{th}$', color='gray', fontsize=12)

# Shade the areas where conditions are met
axs1.axvspan(1, 1.3, facecolor='pink', alpha=0.2)
axs1.axvspan(0, (Lth - tau0) / (2 * r), facecolor='blue', alpha=0.2)

axs1.text(1.05, 40, '$L_n->\infty$', color='red', fontsize=12)
axs1.text(0.1, 40, '$L_0<L_{th}$', color='Blue', fontsize=12)

# Set x-axis limits
axs1.set_xlim(0, 1.3)

# Combine legends from both y-axes
lines, labels = axs1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Add gridlines
axs1.grid(True)

axs1.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size
##
import numpy as np
import matplotlib.pyplot as plt

# Define the constants
L_th = 25
tau_0 = 3
d = 5
M_c = 2
p=0.01
D_0 = p*3 * M_c * d ** 3
# Define different values of g
g_values = [1.5, 1.1, 0.9, 0.6]

# Generate integer vector n from 1 to 20
N = 40
n_values = np.arange(0, N + 1)

# Define different values of alpha
alpha_values = [0.9, 1.0, 1.1]

# Define line styles and colors for different alpha values
line_styles = ['-', '--', ':']
colors = ['#66c2a5', '#fc8d62', '#8da0cb','red']

# Create the plot
plt.figure(figsize=(6, 5))
for alpha_index, alpha in enumerate(alpha_values):
    for g_index, g in enumerate(g_values):
        tau_1 = g / (p * M_c * d ** 2)

        # Initialize an array to store L(n) values
        L_0 = tau_0 + tau_1 * D_0 ** alpha
        L_values = np.zeros(N + 1)
        L_values[0] = L_0

        # Calculate L(n) values based on the provided formula
        for n in n_values[1:]:
            if L_values[n - 1] > L_th:
                L_values[n] = tau_0 + tau_1 * (D_0 + p * (L_values[n - 1] - L_th) * M_c * d ** 2) ** alpha
            else:
                L_values[n] = tau_0 + tau_1 * D_0 ** alpha

        # Use line style and color based on alpha and g
        line_style = line_styles[alpha_index]
        color = colors[g_index]

        plt.plot(n_values, L_values, linestyle=line_style, color=color, label=f'$\\alpha$ = {alpha}, $g$ = {g}')

plt.xlabel('logical gate number n', fontsize=11)
plt.ylabel('feedforward latency $[T_{s}]$', fontsize=11)
plt.grid(True)
plt.yscale('log')  # Set y-axis to log scale
plt.ylim(8, 10 ** 3)
plt.axhline(y=L_th, color='black', linestyle='-.', linewidth=3)
# plt.legend(fontsize=10)  # Adjust the values as needed

plt.show()
##
for alpha_index, alpha in enumerate(alpha_values):
    for g_index, g in enumerate(g_values):
        tau_1 = g / (p * M_c * d ** 2)

        # Initialize an array to store L(n) values
        L_0 = tau_0 + tau_1 * D_0 ** alpha
        L_values = np.zeros(N + 1)
        L_values[0] = L_0
        FLR = np.zeros(N)

        # Calculate L(n) values based on the provided formula
        for n in n_values[1:]:
            if L_values[n - 1] > L_th:
                L_values[n] = tau_0 + tau_1 * (D_0 + p * (L_values[n - 1] - L_th) * M_c * d ** 2) ** alpha
            else:
                L_values[n] = tau_0 + tau_1 * D_0 ** alpha
            FLR[n-1] = L_values[n]/L_values[n-1]
        # Use line style and color based on alpha and g
        line_style = line_styles[alpha_index]
        color = colors[g_index]

        plt.plot(n_values[1:], FLR, linestyle=line_style, color=color, label=f'$\\alpha$ = {alpha}, $g$ = {g}')

plt.xlabel('logical gate number n', fontsize=11)
plt.ylabel('feedforward latency ratio', fontsize=11)
plt.grid(True)
plt.ylim(0.8, 3)
plt.legend(fontsize=10)  # Adjust the values as needed

plt.show()
##
##
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors  # Import the matplotlib.colors module

# Define the constants
L_th = 11
tau_0 = 3
d = 5
M_c = 2
p=0.01
N_0 = p*3 * M_c * d ** 3

n_max = 200

# Create the arrays with evenly spaced values
start_value_g = 0
end_value_g = 1.5
start_value_a = 0.7
end_value_a = 1.5

num_steps = 200
g_values = np.linspace(start_value_g, end_value_g, num_steps)
alpha_values = np.linspace(start_value_a, end_value_a, num_steps*2)

matrix_size = (len(alpha_values), len(g_values))
Latency_ss = np.zeros(matrix_size)
FLR_ss = np.zeros(matrix_size)


for alpha_index, alpha in enumerate(alpha_values):
    for g_index, g in enumerate(g_values):
        tau_1 = g / (p * M_c * d ** 2)

        # Initialize an array to store L(n) values
        L = tau_0 + tau_1 * N_0 ** alpha
        FLR=0
        # Calculate L(n) values based on the provided formula
        for n in np.arange(1, n_max):
            L_temp=L
            if FLR>1000:
                break  # Exit the loop
            if not math.isinf(L):
                if L > L_th:
                    L = tau_0 + tau_1 * (N_0 + p * (L - L_th) * M_c * d ** 2) ** alpha
                else:
                    L = tau_0 + tau_1 * N_0 ** alpha
            FLR = L / L_temp
        Latency_ss[alpha_index,g_index]=L
        FLR_ss[alpha_index,g_index]=FLR
##
# Limit the maximum value in FLR_ss to 5
max_FLR_ss = 20
FLR_ss[FLR_ss > max_FLR_ss] = max_FLR_ss
min_FLR_ss = 0.8
FLR_ss[FLR_ss < min_FLR_ss] = min_FLR_ss
num_color_levels = 100  # Adjust as needed

max_Latency_ss = 10**5.04
Latency_ss[Latency_ss > max_Latency_ss] = max_Latency_ss
min_Latency_ss = L_th
Latency_ss[Latency_ss < min_Latency_ss] = min_Latency_ss

num_color_levels = 100  # Adjust as needed

plt.figure(figsize=(12, 5))  # Adjust the figure size as needed

# Create a subplot for FLR_SS
plt.subplot(1, 3, 1)
plt.contourf(g_values, alpha_values, FLR_ss, cmap='Oranges', vmin=min_FLR_ss, vmax=max_FLR_ss,levels=num_color_levels)  # Set vmax to max_FLR_ss
plt.colorbar()
plt.xlabel('g_values')
plt.ylabel('alpha_values')
plt.title('FLR_SS')

# Create a subplot for latency_ss
plt.subplot(1, 3, 2)
plt.contourf(g_values, alpha_values, np.log10(Latency_ss), cmap='RdYlBu_r', vmin=np.log10(min_Latency_ss), vmax=np.log10(max_Latency_ss), levels=num_color_levels)
plt.colorbar()
plt.xlabel('g_values')
plt.ylabel('alpha_values')
plt.title('latency_ss')

plt.subplot(1, 3, 3)  # Add a third subplot
alpha_1_index = np.argmin(np.abs(alpha_values - 1))  # Find the index where alpha is closest to 1
latency_ss_alpha_1 = Latency_ss[alpha_1_index, :] # Extract data for alpha = 1
plt.plot(g_values, latency_ss_alpha_1, color='b', label='latency_ss for alpha = 1')
plt.xlabel('g_values')
plt.ylabel('Latency_ss')
plt.title('Latency_ss vs. g_values')

plt.grid(True)
plt.ylim(0, 100)


plt.tight_layout()
plt.show()

##

import numpy as np
import matplotlib.pyplot as plt

L_th = 11
tau_0 = 3
d = 5
M_c = 2
p = 0.01
N_0 = p * (3 * d) * M_c * d ** 2


# Define the specified values for 'g' and 'alpha'
g_values = [1.6, 1.1, 0.9, 0.6]
alpha_values = [0.9, 1.0, 1.1]

fig, axs = plt.subplots(1,len(alpha_values), figsize=(4 * len(alpha_values),3))

for i, alpha in enumerate(alpha_values):
    for g in g_values:
        tau_1 = g / (p * M_c * d ** 2)
        N = np.linspace(0, 5 * N_0, 1000)
        L = np.linspace(0, 5 * L_th, 1000)
        L_dec = (tau_0 + tau_1 * N ** alpha)
        N_L = N_0+(p * M_c * d ** 2)*(L-L_th)* np.heaviside(L-L_th, N_0)

        L_ff = L_th + (N - N_0) * tau_1 / g * np.heaviside(N - N_0, L_th)

        # Plot on the current subplot
        axs[i].plot(N/N_0, L_dec/L_th , label=f'g={g}')
        axs[i].set_title(f'alpha={alpha}')
        axs[i].legend()
    axs[i].plot(N_L/N_0 , L/L_th )
    axs[i].set_ylim([0,8])

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

##import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors  # Import the matplotlib.colors module

# Define the constants
L_th = 11
tau_0 = 1
d = 5
M_c = 2
p=0.01
N_0 = p*3 * M_c * d ** 3

n_max = 200

# Create the arrays with evenly spaced values
start_value_g = 0
end_value_g = 1.5
start_value_a = 0.7
end_value_a = 1.5

num_steps = 200
g_values = np.linspace(start_value_g, end_value_g, num_steps)
alpha_values = np.linspace(start_value_a, end_value_a, num_steps*2)

matrix_size = (len(alpha_values), len(g_values))
Latency_ss = np.zeros(matrix_size)
FLR_ss = np.zeros(matrix_size)
throughput_ss=np.zeros(matrix_size)
arrival = (p * M_c * d ** 2)

for alpha_index, alpha in enumerate(alpha_values):
    for g_index, g in enumerate(g_values):
        tau_1 = g / (p * M_c * d ** 2)
        # Initialize an array to store L(n) values
        L = tau_0 + tau_1 * N_0 ** alpha
        FLR=0
        # Calculate L(n) values based on the provided formula
        for n in np.arange(1, n_max):
            L_temp=L
            if FLR>1000:
                break  # Exit the loop
            if not math.isinf(L):
                L = tau_0 + tau_1 * (N_0 + (L - L_th) * arrival) ** alpha
                if L < L_th:
                    L = tau_0 + tau_1 * N_0 ** alpha
            FLR = L / L_temp
        Latency_ss[alpha_index,g_index]=L
        FLR_ss[alpha_index,g_index]=FLR
        throughput_ss[alpha_index,g_index]=(alpha*tau_1*(N_0 + (L - L_th) * arrival*int(L>L_th))**(alpha-1))*arrival
##
# Limit the maximum value in FLR_ss to 5
max_FLR_ss = 20
FLR_ss[FLR_ss > max_FLR_ss] = max_FLR_ss
min_FLR_ss = 0.8
FLR_ss[FLR_ss < min_FLR_ss] = min_FLR_ss
num_color_levels = 100  # Adjust as needed

max_Latency_ss = 10**5.04
Latency_ss[Latency_ss > max_Latency_ss] = max_Latency_ss
min_Latency_ss = L_th
Latency_ss[Latency_ss < min_Latency_ss] = min_Latency_ss

max_throughput_ss = 10**2
throughput_ss[throughput_ss > max_throughput_ss] = max_throughput_ss
min_throughput_ss = 10**-3
throughput_ss[throughput_ss < min_throughput_ss] = min_throughput_ss


num_color_levels = 100  # Adjust as needed

plt.figure(figsize=(12, 5))  # Adjust the figure size as needed

# Create a subplot for FLR_SS
plt.subplot(1, 3, 1)
plt.contourf(g_values, alpha_values, FLR_ss, cmap='Oranges', vmin=min_FLR_ss, vmax=max_FLR_ss,levels=num_color_levels)  # Set vmax to max_FLR_ss
plt.colorbar()
plt.xlabel('g_values')
plt.ylabel('alpha_values')
plt.title('FLR_SS')

# Create a subplot for latency_ss
plt.subplot(1, 3, 2)
plt.contourf(g_values, alpha_values, np.log10(throughput_ss), cmap='RdYlBu_r', vmin=np.log10(min_throughput_ss), vmax=np.log10(max_throughput_ss), levels=num_color_levels)
plt.colorbar()
plt.xlabel('linear utilization')
plt.ylabel('alpha_values')
plt.title('steady-state Utilization')

plt.subplot(1, 3, 3)  # Add a third subplot
alpha_1_index = np.argmin(np.abs(alpha_values - 1))  # Find the index where alpha is closest to 1
latency_ss_alpha_1 = Latency_ss[alpha_1_index, :] # Extract data for alpha = 1
plt.plot(g_values, latency_ss_alpha_1, color='b', label='latency_ss for alpha = 1')
plt.xlabel('g_values')
plt.ylabel('Latency_ss')
plt.title('Latency_ss vs. g_values')

plt.grid(True)
plt.ylim(0, 100)


plt.tight_layout()
plt.show()
