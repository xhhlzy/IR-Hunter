import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(
        description="Load a NumPy .npy file and display it as a color image"
    )
    parser.add_argument(
        "npy_file",
        help="Path to the .npy file to visualize"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npy_file):
        print(f"Error: File not found: {args.npy_file}", file=sys.stderr)
        sys.exit(1)

    # 加载并 squeeze
    data = np.load(args.npy_file)
    data = np.squeeze(data)

    # 只考虑非零的像素来计算 pct 百分位
    flat = data.flatten()
    non_zero = flat[flat > 0]
    if non_zero.size == 0:
        print("Warning: all data are zero.", file=sys.stderr)
        vmin, vmax = 0, 1
    else:
        # 取 2%–98% 百分位作为下/上界
        vmin, vmax = np.percentile(non_zero, [2, 98])

    # 准备 colormap，under-color 设成示例里的浅青
    cmap = plt.get_cmap("jet")
    cmap.set_under("#000085")   # 浅青，和示例背景一致
    cmap.set_over("red")        # 超过 vmax 的用红色（可选）
    cmap.set_bad("lightgray")   # 若有 NaN，用浅灰（可选）

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    # 绘图
    plt.figure(figsize=(10, 8))
    if data.ndim == 3 and data.shape[2] in (3, 4):
        plt.imshow(data, origin="lower", interpolation="nearest")
    else:
        plt.imshow(
            data,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest"
        )

    # colorbar
    cbar = plt.colorbar(
        extend="both",
        pad=0.02,
        aspect=30
    )
    cbar.set_label("Value", rotation=270, labelpad=15)

    plt.title(f"Visualization of {os.path.basename(args.npy_file)}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
