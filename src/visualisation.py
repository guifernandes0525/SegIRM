import matplotlib.pyplot as plt
from adjustText import adjust_text

# -------------------------
# DATA
# -------------------------

slic_nreal = [39, 79, 183, 268, 448, 746]
slic_br = [0.71204, 0.81761, 0.92212, 0.94373, 0.98803, 0.99875]
slic_ue = [0.49921, 0.43814, 0.33774, 0.28919, 0.24594, 0.20226]

felz_nreal = [224, 131, 102, 69]
felz_br = [0.95606, 0.92479, 0.91979, 0.85995]
felz_ue = [0.29627, 0.36967, 0.38550, 0.45491]
felz_scale = [50, 100, 150, 200]

watershed_nreal = [49, 94, 191, 298, 531]
watershed_br = [0.46445, 0.62522, 0.80368, 0.87478, 0.95248]
watershed_ue = [0.56003, 0.52501, 0.40296, 0.37045, 0.28935]

# -------------------------
# PLOT
# -------------------------

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(slic_ue, slic_br, color="red", s=60, label="SLIC (c=0.05)", zorder=3)
ax.scatter(felz_ue, felz_br, color="blue", s=60, label="Felzenszwalb (σ=0.5, min=20)", zorder=3)
ax.scatter(watershed_ue, watershed_br, color="green", s=60, label="Watershed", zorder=3)

texts = []
for i, txt in enumerate(slic_nreal):
    texts.append(ax.text(slic_ue[i], slic_br[i], f"n={txt}", fontsize=8, color="red"))
for i, txt in enumerate(felz_nreal):
    texts.append(ax.text(felz_ue[i], felz_br[i], f"n={txt}(s={felz_scale[i]})", fontsize=8, color="blue"))
for i, txt in enumerate(watershed_nreal):
    texts.append(ax.text(watershed_ue[i], watershed_br[i], f"n={txt}", fontsize=8, color="green"))

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

ax.set_xlabel("Undersegmentation Error (UE)")
ax.set_ylabel("Boundary Recall (BR)")
ax.set_title("Segmentation Algorithm Comparison (BR vs UE)")
ax.grid(True)
ax.legend()
ax.set_xlim(0, 0.65)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()