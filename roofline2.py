import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
from matplotlib.ticker import LogLocator, ScalarFormatter

# Load into DataFrame
df = pd.read_csv("benchmarks.tsv", sep="\t")

# Compute Arithmetic Intensity (GFLOPs / (size^2 * sizeof(float) * 3) in GB)
# Total bytes = 3 matrices of size N^2 * 4 bytes = 12*N^2 bytes = 12*N^2/1e9 GB
df['AI'] = (2* df['Size']**3) / (3 * df['Size']**2 * 4 / 1e9)
print(df['AI'])
# Roofline parameters
peak_flops = 56*4  # GFLOPs
mem_bw = 76.8  # GB/s

# Generate AI range
ai_range = np.logspace(-1, 12, 100)
roofline = np.minimum(peak_flops, ai_range * mem_bw)

# Plot
plt.figure(figsize=(10, 7))
plt.loglog(ai_range, roofline, '--', color='black', label='Roofline')
plt.axhline(peak_flops, color='red', linestyle='--', label='Peak FLOPs')

# Plot each point
colors = sns.color_palette("tab20", n_colors=17)
naiive_mask = df["Kernel"].str.lower().str.contains("naiive")

for i, (name, group) in enumerate(df.groupby("Kernel")):
    style = 'o' if not naiive_mask[group.index].any() else 'o'
    edgecolor = 'black' if naiive_mask[group.index].any() else None
    plt.scatter(group["AI"], group["GFLOPs"], label=name, s=60, marker=style, edgecolors=edgecolor, color=colors[i % 14])

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOPs/s)")
plt.title("Roofline Plot")
plt.legend(loc="lower left", fontsize="small", ncol=2)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.xscale("log")
plt.yscale("log")
plt.show()

