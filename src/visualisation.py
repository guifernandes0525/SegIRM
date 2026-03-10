import matplotlib.pyplot as plt

# -------------------------
# DATA
# -------------------------

# SLIC
slic_n = [50,100,200,300,500,800]
slic_br = [0.466,0.616,0.785,0.888,0.980,0.999]
slic_ue = [0.540,0.450,0.370,0.314,0.263,0.224]

# Felzenszwalb
felz_scale = [50,100,150,200,300,500]
felz_br = [0.883,0.858,0.778,0.756,0.239,0.097]
felz_ue = [0.197,0.268,0.269,0.279,0.406,0.409]

# Watershed
watershed_markers = [50,100,150,200,300,500]
watershed_br = [0.550,0.726,0.836,0.879,0.954,0.991]
watershed_ue = [0.227,0.183,0.155,0.139,0.119,0.0098]


# -------------------------
# PLOT
# -------------------------

plt.figure(figsize=(8,6))

# SLIC
plt.scatter(slic_ue, slic_br, color="red", label="SLIC")
for i, txt in enumerate(slic_n):
    plt.text(slic_ue[i]+0.005, slic_br[i], f"n={txt}", fontsize=9)

# Felzenszwalb
plt.scatter(felz_ue, felz_br, color="blue", label="Felzenszwalb")
for i, txt in enumerate(felz_scale):
    plt.text(felz_ue[i]+0.005, felz_br[i], f"s={txt}", fontsize=9)

# Watershed
plt.scatter(watershed_ue, watershed_br, color="green", label="Watershed")
for i, txt in enumerate(watershed_markers):
    plt.text(watershed_ue[i]+0.005, watershed_br[i], f"m={txt}", fontsize=9)


# -------------------------
# GRAPH SETTINGS
# -------------------------

plt.xlabel("Undersegmentation Error (UE)")
plt.ylabel("Boundary Recall (BR)")
plt.title("Segmentation Algorithm Comparison")

plt.grid(True)
plt.legend()

plt.xlim(0,0.6)
plt.ylim(0,1.05)

plt.show()

