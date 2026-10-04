"""
Step 4, Plot B: the EFT-motivated stress test.

Instead of a constant xi, use xi(m) = 1 + c*(m/1 TeV)^2 -- a toy
momentum-transfer-dependent coupling, the kind of energy dependence a
dimension-6 EFT operator can introduce. This directly targets the
assumption named in the ATLAS off-shell paper (arXiv:1808.01191):
"Assuming the ratio of the Higgs boson couplings to the Standard Model
predictions is independent of the momentum transfer..."

The point of this plot is NOT just a deeper dip (that's what uniform xi
does, see plot_kappa_scan.py) -- it's a genuine SHAPE difference: the
high-mass tail moves more than the near-peak region, because xi(m)
itself grows with mass.

Usage:
    python3 plot_eft_stress_test.py lineshape_data.csv
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

from coupling_reweight import load_data, xi_eft_toy, xi_uniform, reweighted_pieces


def main(csv_path):
    df = load_data(csv_path)

    c_values = [-0.5, 0.0, 0.5]
    colors = ["#4c72b0", "#000000", "#c44e52"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for c, color in zip(c_values, colors):
        signal, interf, _, _ = reweighted_pieces(df, xi_eft_toy, c=c)
        higgs_related = signal + interf
        lw = 2.2 if c == 0.0 else 1.4
        label = rf"$c = {c}$" if c != 0.0 else r"$c = 0$  (SM baseline)"
        ax.plot(
            df["mass_mid_GeV"], higgs_related,
            marker="o", markersize=4, linewidth=lw, color=color, label=label,
        )

    # for reference, show what a uniform xi=1.3 would look like (same
    # on-peak normalization change as the c=0.5 curve gets, roughly, but
    # WITHOUT the energy dependence) to make the shape difference explicit
    signal_u, interf_u, _, _ = reweighted_pieces(df, xi_uniform, xi_value=1.10)
    ax.plot(
        df["mass_mid_GeV"], signal_u + interf_u,
        marker="none", linewidth=1.2, linestyle="--", color="gray",
        label=r"uniform $\xi=1.10$ (for shape comparison)",
    )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(125.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(r"$\sigma^{H+I}$ [fb]")
    ax.set_title(r"Higgs-related lineshape: $\xi(m)=1+c\,(m/1\,\mathrm{TeV})^2$ toy EFT coupling")
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

    out_pdf = "eft_stress_test.pdf"
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lineshape_data.csv"
    main(path)
