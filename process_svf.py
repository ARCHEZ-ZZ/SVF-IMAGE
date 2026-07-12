import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 1) FILE NAMES
# =========================
excel_file = "SVF EXCEL.xlsx"
base_image_file = "BASE.png"
color_reference_file = "COLOR.jpg"
output_file = "SVF_colored_result.png"

# =========================
# 2) READ EXCEL
# =========================
# This assumes the Excel file contains a 400x400 grid of SVF values
df = pd.read_excel(excel_file, header=None)

# If there are extra empty rows/columns, keep only first 400x400
svf = df.values[:400, :400].astype(float)

# =========================
# 3) NORMALIZE VALUES
# =========================
svf_min = np.nanmin(svf)
svf_max = np.nanmax(svf)

# Avoid division by zero
if svf_max == svf_min:
    norm_svf = np.zeros_like(svf)
else:
    norm_svf = (svf - svf_min) / (svf_max - svf_min)

# =========================
# 4) CREATE A COLOR MAP
# =========================
# Option A: built-in scientific colormap
# cmap = plt.get_cmap("jet")   # strong rainbow style
# cmap = plt.get_cmap("viridis")  # smooth scientific style
# cmap = plt.get_cmap("turbo")    # vivid style

# Option B: custom from 3 colors (blue -> green -> red)
cmap = LinearSegmentedColormap.from_list(
    "custom_svf",
    ["blue", "cyan", "yellow", "red"],
    N=256
)

# Convert normalized data to RGBA
colored_rgba = cmap(norm_svf)

# Convert to 8-bit RGB
colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

# =========================
# 5) LOAD BASE IMAGE
# =========================
base_img = Image.open(base_image_file).convert("RGBA")
base_img = base_img.resize((400, 400), Image.Resampling.LANCZOS)
base_arr = np.array(base_img)

# =========================
# 6) OVERLAY COLOR MAP ON BASE IMAGE
# =========================
# Make the SVF layer semi-transparent
alpha = 0.75  # change between 0 and 1
svf_layer = np.dstack([
    colored_rgb,
    (np.ones((400, 400)) * 255 * alpha).astype(np.uint8)
])

# Blend base image with SVF colored layer
base_rgb = base_arr[:, :, :3].astype(np.float32)
svf_rgb = colored_rgb.astype(np.float32)

blended_rgb = (0.35 * base_rgb + 0.65 * svf_rgb).astype(np.uint8)

# =========================
# 7) REMOVE BORDER / GRID LINES
# =========================
# This is a simple cleanup step.
# If your border lines are black/dark, this tries to replace very dark pixels
# with nearby average color.
result = blended_rgb.copy()

gray = np.mean(result, axis=2)
dark_mask = gray < 40  # adjust threshold if needed

# Replace dark pixels with neighboring average by simple smoothing
from scipy.ndimage import median_filter
smoothed = median_filter(result, size=(3, 3, 1))
result[dark_mask] = smoothed[dark_mask]

# =========================
# 8) SAVE OUTPUT
# =========================
out_img = Image.fromarray(result)
out_img.save(output_file)

print(f"Saved final image as: {output_file}")
