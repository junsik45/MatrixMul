import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Data
memory_bandwidth = 5.64  # GB/s
flops_per_byte = 11.6  # Flops/byte
peak_flops = 65.4  # GFlops/s

size = 512
#AI         = 166.67 # Flops/byte
AI = 0.
if size == 2048:
    AI = 166.67
elif size == 1024:
    AI = 166.67/2
elif size == 512:
    AI = 166.67/4


benchmark = {}
with open('benchmarks', 'r') as file:
    for line in file:
        key, value = line.strip().split()
        benchmark[key] = value
print(benchmark)

# Points for the roofline
x_whole = np.linspace(flops_per_byte, 200, 100)  # Bandwidth in GB/s
x_memory = np.linspace(0,flops_per_byte, 100)  # Bandwidth in GB/s
y_memory = memory_bandwidth * x_memory  # Roofline for memory bandwidth
y_peak = np.full_like(x_whole, peak_flops)  # Peak performance line

# Create the plot
fig, ax = plt.subplots()

# Plot memory bandwidth line
ax.plot(x_memory, y_memory, color='blue')
ax.plot(x_whole, y_peak, color='red', linestyle='--')

# Indicating the balance point
ax.scatter([flops_per_byte], [peak_flops], color='green')
plt.text(flops_per_byte + 3, peak_flops, 'Balance Point', fontsize=9, color='green')

# Indicating individual point
for method in benchmark.keys():
    ax.plot([AI], [float(benchmark[method])], marker='o', mfc='none', markersize=12, label=method)

# Labels and title
ax.set_xticks(ticks=[10, 30, 50, 70])  # Omit tick at 0
plt.xlim(0, 100)
plt.ylim(0, 75)  # Set a reasonable y limit for visualization
plt.xlabel('Arithmetic Intensity (Flops/byte)')
plt.ylabel('Performance (GFlops/s)')
plt.title('Roofline Model')
plt.grid()
plt.legend()
plt.tight_layout()
# Show the plot
plt.savefig('roofline.pdf')

