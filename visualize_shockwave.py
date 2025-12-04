import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

csv_file = "output/shockwave_matrix.csv"
data = np.loadtxt(csv_file, delimiter=",")

data[data <= 0] = 1e-3

plt.figure(figsize=(10, 8))

plt.imshow(
    data,
    cmap="inferno",
    origin="lower",
    interpolation="nearest",
    norm=LogNorm()
)

plt.colorbar(label="Overpressure (kPa, Log Scale)")
plt.title("Shockwave Overpressure (2D Heatmap — Log Color Scale)")
plt.xlabel("X (cells)")
plt.ylabel("Y (cells)")

plt.tight_layout()
plt.show()
