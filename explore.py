import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ---------- FLAIR scan ----------
path = "data/BraTS20_Training_001/BraTS20_Training_001_flair.nii"

img = nib.load(path)
data = img.get_fdata()

print("Shape:", data.shape)
print("Min:", data.min())
print("Max:", data.max())

# ---------- Seg (mask) file ----------                          # NEW
seg_path = "data/BraTS20_Training_001/BraTS20_Training_001_seg.nii" # NEW
seg_data = nib.load(seg_path).get_fdata()                         # NEW
print("Seg labels:", np.unique(seg_data))                         # NEW

# ---------- Slice kaattu ----------
slice_2d = data[:, :, 77].T                                       # NEW: .T
plt.imshow(slice_2d, cmap="gray", origin="lower")                 # NEW: origin
plt.title("FLAIR - slice 77")
plt.show()