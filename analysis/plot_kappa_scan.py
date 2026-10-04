"""
Step 4, Plot A: the standard (energy-independent) coupling rescaling scan.

rerun with the Higgs coupling
hand-scaled by a constant xi (0.8, 1.0, 1.2, 1.5) and show how the dip
and high-mass excess move -- entirely by reweighting the already-computed
128/129/132 lineshape, using the xi^4 / xi^2 / unchanged scaling derived
in coupling_reweight.py.

Usage:
    python3 plot_kappa_scan.py lineshape_data.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # Times-like math glyphs to match
plt.rcParams["font.size"] = 14          # base size; axis labels, ticks, legend all scale from this
# plt.rcParams["axes.titlesize"] = 14     # plot title
# plt.rcParams["axes.labelsize"] = 14     # x/y axis labels
# plt.rcParams["xtick.labelsize"] = 12    # x-axis tick numbers
# plt.rcParams["ytick.labelsize"] = 12    # y-axis tick numbers
plt.rcParams["legend.fontsize"] = 12    # legend text

from coupling_reweight import load_data, xi_uniform, reweighted_pieces


def main(csv_path):
    df = load_data(csv_path)
    xi_values = [0.8, 1.0, 1.2, 1.5]
    colors = ["#4c72b0", "#000000", "#dd8452", "#c44e52"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for xi, color in zip(xi_values, colors):
        signal, interf, _, _ = reweighted_pieces(df, xi_uniform, xi_value=xi)
        higgs_related = signal + interf
        lw = 2.2 if xi == 1.0 else 1.4
        ax.plot(
            df["mass_mid_GeV"], higgs_related,
            marker="o", markersize=4, linewidth=lw, color=color,
            label=rf"$\xi = {xi}$" + ("  (SM baseline)" if xi == 1.0 else ""),
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(125.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(r"$\sigma^{H+I} = |M_H+M_C|^2-|M_C|^2$ [fb]")
    ax.set_title(r"Higgs-related lineshape under a uniform coupling rescaling $\xi$")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    # nudge the "-1e-6"-scale tick label down so it doesn't clash with "0"
    fig.canvas.draw()
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if tick < 0 and abs(tick) < 5e-6:
            label.set_verticalalignment("top")
            label.set_transform(
            label.get_transform()
            + ScaledTranslation(0, -5/72, fig.dpi_scale_trans)
        )

    out_pdf = "kappa_scan.pdf"
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lineshape_data.csv"
    main(path)
