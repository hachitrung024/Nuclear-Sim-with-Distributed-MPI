import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file CSV
# path1 = "data/radioactive_matrix.csv"  # Hình ảnh đầu tiên
# path2 = "output/radioactive_matrix.csv"  # Hình ảnh thứ hai (có thể thay đổi đường dẫn)
path1 = "data/heat_matrix.csv"  # Hình ảnh đầu tiên
path2 = "output/heat_matrix.csv"  # Hình ảnh thứ hai (có thể thay đổi đường dẫn)

# Đọc dữ liệu của hai hình ảnh
matrix1 = pd.read_csv(path1, header=None).values
matrix2 = pd.read_csv(path2, header=None).values

# Tạo hình ảnh với hai subplot để so sánh
fig, axes = plt.subplots(1, 2, figsize=(14, 8))  # 1 hàng, 2 cột

# Vẽ heatmap đầu tiên và thêm colorbar
im1 = axes[0].imshow(matrix1, cmap="viridis", interpolation="nearest")
axes[0].set_title("Radioactive Contamination Heatmap 1 (viridis)")
axes[0].set_xlabel("X position")
axes[0].set_ylabel("Y position")
fig.colorbar(im1, ax=axes[0], label="Radioactive Level")

# Vẽ heatmap thứ hai và thêm colorbar
im2 = axes[1].imshow(matrix2, cmap="viridis", interpolation="nearest")
axes[1].set_title("Radioactive Contamination Heatmap 2 (viridis)")
axes[1].set_xlabel("X position")
axes[1].set_ylabel("Y position")
fig.colorbar(im2, ax=axes[1], label="Radioactive Level")

# Hiển thị kết quả
plt.tight_layout()
plt.show()
