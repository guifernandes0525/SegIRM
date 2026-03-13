import matplotlib.pyplot as plt

# -------------------------
# DATA
# -------------------------

# SLIC
slic_segments = [50,100,200,300,500,800]
slic_ue  = [0.53984,0.45019,0.37035,0.31430,0.26313,0.22434]
slic_br  = [0.46552,0.61593,0.78546,0.88764,0.98035,0.99946]
slic_src = [0.90356,0.90920,0.91385,0.90976,0.89894,0.88118]

# Watershed
watershed_segments = [50,100,200,300,500]
watershed_ue  = [0.55673,0.52726,0.41234,0.37302,0.29580]
watershed_br  = [0.46570,0.61611,0.79671,0.87299,0.95177]
watershed_src = [0.38216,0.39979,0.49904,0.46166,0.52105]

# Felzenszwalb (sigma=0.5 , min_size=20)
felz_scale = [50,100,200,300]
felz_ue  = [0.31221,0.38031,0.45958,0.49345]
felz_br  = [0.95355,0.92176,0.83548,0.74062]
felz_src = [0.36806,0.28407,0.22211,0.17692]

# -------------------------
# FIGURE AVEC 3 SUBPLOTS
# -------------------------

fig, axs = plt.subplots(1,3, figsize=(18,5))

#Analyse Error

axs[0].scatter(slic_segments, slic_ue, label="SLIC")
axs[0].scatter(watershed_segments, watershed_ue, label="Watershed")
axs[0].scatter(felz_scale, felz_ue, label="Felzenszwalb")

axs[0].set_xlabel("Number of segments / markers / scale")
axs[0].set_ylabel("Undersegmentation Error (UE)")
axs[0].set_title("UE vs Segments")
axs[0].grid(True)

#Analyse Boundary Recall

axs[1].scatter(slic_segments, slic_br, label="SLIC")
axs[1].scatter(watershed_segments, watershed_br, label="Watershed")
axs[1].scatter(felz_scale, felz_br, label="Felzenszwalb")

axs[1].set_xlabel("Number of segments / markers / scale")
axs[1].set_ylabel("Boundary Recall (BR)")
axs[1].set_title("BR vs Segments")
axs[1].grid(True)

#Analyse régularité 

axs[2].scatter(slic_segments, slic_src, label="SLIC")
axs[2].scatter(watershed_segments, watershed_src, label="Watershed")
axs[2].scatter(felz_scale, felz_src, label="Felzenszwalb")

axs[2].set_xlabel("Number of segments / markers / scale")
axs[2].set_ylabel("Shape Regularity Criterion (SRC)")
axs[2].set_title("SRC vs Segments")
axs[2].grid(True)

# -------------------------
# LEGEND UNIQUE
# -------------------------

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

plt.tight_layout()
plt.show()