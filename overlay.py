import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

patient = "BraTS20_Training_001"
folder = f"data/{patient}"
z = 77

flair = nib.load(f"{folder}/{patient}_flair.nii").get_fdata()
seg = nib.load(f"{folder}/{patient}_seg.nii").get_fdata()

# ---------- Muzhu 3D brain counts ----------                    # NEW
necrotic  = np.sum(seg == 1)                                      # NEW
oedema    = np.sum(seg == 2)                                      # NEW
enhancing = np.sum(seg == 4)                                      # NEW
total_tumour = necrotic + oedema + enhancing                      # NEW
total_voxels = seg.size                                           # NEW: 240*240*155

print("=== Whole 3D brain ===")                                   # NEW
print("Necrotic voxels :", necrotic)                              # NEW
print("Oedema voxels   :", oedema)                                # NEW
print("Enhancing voxels:", enhancing)                             # NEW
print("Total tumour    :", total_tumour)                          # NEW
print("Total voxels    :", total_voxels)                          # NEW
print(f"Tumour %        : {total_tumour / total_voxels * 100:.2f} %")  # NEW

# ---------- Oru slice ----------
img = flair[:, :, z].T
mask = seg[:, :, z].T

print(f"\n=== Slice z = {z} ===")                                 # NEW
print("Necrotic :", np.sum(mask == 1))                            # NEW (mask ready aana pinnaadi!)
print("Oedema   :", np.sum(mask == 2))                            # NEW
print("Enhancing:", np.sum(mask == 4))                            # NEW

tumour = np.ma.masked_where(mask == 0, mask)     # 0 → transparent

colours = ListedColormap(["red", "green", "yellow"])
norm = BoundaryNorm([0.5, 1.5, 3, 4.5], colours.N)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Original scan
axes[0].imshow(img, cmap="gray", origin="lower")
axes[0].set_title("FLAIR")

# 2. Mask mattum
axes[1].set_facecolor("black")
axes[1].imshow(tumour, cmap=colours, norm=norm, origin="lower")
axes[1].set_title("Tumour mask")

# 3. Scan + mask overlay
axes[2].imshow(img, cmap="gray", origin="lower")
axes[2].imshow(tumour, cmap=colours, norm=norm, alpha=0.5, origin="lower")
axes[2].set_title(f"Overlay (z = {z})")                           # NEW: z title-la

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

legend = [Patch(color="red", label="1: Necrotic core"),
          Patch(color="green", label="2: Oedema"),
          Patch(color="yellow", label="4: Enhancing tumour")]
fig.legend(handles=legend, loc="lower center", ncol=3)

plt.show()