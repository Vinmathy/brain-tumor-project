import sys
import time
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
from matplotlib.patches import Patch
from model import UNet3D

# ---------- Settings (Colab-la use pannadhe) ----------
MODALITIES = ["flair", "t1", "t1ce", "t2"]
X0, X1, Y0, Y1, Z0, Z1 = 40, 200, 24, 216, 8, 152
REGIONS = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}

def normalize(vol):
    brain = vol > 0
    out = np.zeros_like(vol, dtype=np.float32)
    out[brain] = (vol[brain] - vol[brain].mean()) / vol[brain].std()
    return out

def load_patient(p):
    folder = f"data/{p}"
    scans = [normalize(nib.load(f"{folder}/{p}_{m}.nii").get_fdata()) for m in MODALITIES]
    image = np.stack(scans, axis=0)[:, X0:X1, Y0:Y1, Z0:Z1]
    seg = nib.load(f"{folder}/{p}_seg.nii").get_fdata().astype(np.uint8)
    seg[seg == 4] = 3
    return image, seg[X0:X1, Y0:Y1, Z0:Z1]

def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2 * inter / total

# ---------- 1. Load patient ----------
patient = sys.argv[1] if len(sys.argv) > 1 else "BraTS20_Training_001"
print(f"Loading {patient} ...")
image, gt = load_patient(patient)

# ---------- 2. Model prediction (CPU) ----------
model = UNet3D()
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()

t0 = time.time()
with torch.no_grad():
    logits = model(torch.from_numpy(image).unsqueeze(0))
pred = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
print(f"Prediction time (CPU): {time.time() - t0:.1f} s")

scores = {r: dice_score(np.isin(pred, l), np.isin(gt, l)) for r, l in REGIONS.items()}
print("Dice:", {r: round(float(s), 3) for r, s in scores.items()})

# ---------- 3. Viewer ----------
flair = image[0]
colours = ListedColormap(["red", "green", "yellow"])
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], colours.N)
mask = lambda s: np.ma.masked_where(s.T == 0, s.T)        # tumour: label 0 → transparent
brain = lambda s: np.ma.masked_where(s.T == 0, s.T)       # NEW: background → transparent

z0 = int(np.argmax((gt > 0).sum(axis=(0, 1))))

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(bottom=0.2)

for ax in axes:                                           # NEW
    ax.set_facecolor("black")                             # NEW: transparent idam karuppu

bg = [ax.imshow(brain(flair[:, :, z0]), cmap="gray", origin="lower") for ax in axes]  # NEW
gt_im = axes[1].imshow(mask(gt[:, :, z0]), cmap=colours, norm=norm, alpha=0.5, origin="lower")
pr_im = axes[2].imshow(mask(pred[:, :, z0]), cmap=colours, norm=norm, alpha=0.5, origin="lower")

axes[0].set_title(f"FLAIR (z = {z0})")                    # NEW: z title-la
axes[1].set_title("Doctor (ground truth)")
axes[2].set_title(f"Model  |  WT {scores['WT']:.2f}  TC {scores['TC']:.2f}  ET {scores['ET']:.2f}")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(patient)
fig.legend(handles=[Patch(color="red", label="Necrotic"),
                    Patch(color="green", label="Oedema"),
                    Patch(color="yellow", label="Enhancing")],
           loc="upper right")

slider_ax = fig.add_axes([0.2, 0.06, 0.6, 0.03])
slider = Slider(slider_ax, "Slice z", 0, flair.shape[2] - 1, valinit=z0, valstep=1)

def update(v):
    z = int(v)
    for im in bg:
        im.set_data(brain(flair[:, :, z]))                # NEW
    gt_im.set_data(mask(gt[:, :, z]))
    pr_im.set_data(mask(pred[:, :, z]))
    axes[0].set_title(f"FLAIR (z = {z})")                 # NEW
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()