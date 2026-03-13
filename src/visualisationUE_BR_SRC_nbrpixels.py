import matplotlib.pyplot as plt

# -------------------------
# DATA — NSegments_real en X
# -------------------------

# SLIC (compactness=0.05, sigma=1)
slic_segments = [39, 79, 183, 268, 448, 746]
slic_ue  = [0.49921, 0.43814, 0.33774, 0.28919, 0.24594, 0.20226]
slic_br  = [0.71204, 0.81761, 0.92212, 0.94373, 0.98803, 0.99875]
slic_src = [0.38242, 0.44956, 0.57692, 0.66413, 0.71686, 0.75577]

# Watershed
watershed_segments = [49, 94, 191, 298, 531]
watershed_ue  = [0.56003, 0.52501, 0.40296, 0.37045, 0.28935]
watershed_br  = [0.46445, 0.62522, 0.80368, 0.87478, 0.95248]
watershed_src = [0.37384, 0.39079, 0.48731, 0.45438, 0.51866]

# Felzenszwalb (sigma=0.5, min_size=20)
felz_segments = [224, 131, 102, 69]
felz_ue  = [0.29627, 0.36967, 0.38550, 0.45491]
felz_br  = [0.95606, 0.92479, 0.91979, 0.85995]
felz_src = [0.37898, 0.27844, 0.25661, 0.21493]

# -------------------------
# FIGURE AVEC 3 SUBPLOTS
# -------------------------

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ax, y_slic, y_ws, y_felz, ylabel, title in zip(
    axs,
    [slic_ue, slic_br, slic_src],
    [watershed_ue, watershed_br, watershed_src],
    [felz_ue, felz_br, felz_src],
    ["Undersegmentation Error (UE)", "Boundary Recall (BR)", "Shape Regularity Criterion (SRC)"],
    ["UE vs NSegments réels", "BR vs NSegments réels", "SRC vs NSegments réels"]
):
    ax.scatter(slic_segments, y_slic, label="SLIC (c=0.05)")
    ax.scatter(watershed_segments, y_ws, label="Watershed")
    ax.scatter(felz_segments, y_felz, label="Felzenszwalb (σ=0.5, min=20)")
    ax.set_xlabel("NSegments réels")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()