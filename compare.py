import nibabel as nib
import matplotlib.pyplot as plt

patient = "BraTS20_Training_001"
folder = f"data/{patient}"
modalities = ["flair", "t1", "t1ce", "t2"]
z = 77

fig, axes = plt.subplots(1, 4, figsize=(16, 4))   # 1 row, 4 frames

for i, m in enumerate(modalities):
    path = f"{folder}/{patient}_{m}.nii"          # e.g. ..._t1ce.nii
    data = nib.load(path).get_fdata()
    slice_2d = data[:, :, z].T

    axes[i].imshow(slice_2d, cmap="gray", origin="lower")
    axes[i].set_title(m.upper())                  # "flair" → "FLAIR"
    axes[i].axis("off")                           # numbers/axis kodugal vendaam

plt.tight_layout()                                # frames-ukku naduvula idaiveli sari pannum
plt.show()