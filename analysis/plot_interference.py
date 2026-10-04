"""
Plot the gg -> H* -> ZZ interference lineshape (MCFM process 129)
as a function of the four-lepton invariant mass.

Usage:
    python3 plot_interference.py interference_lineshape.csv
"""

import sys
import pandas as pd
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


def main(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("mass_mid_GeV").reset_index(drop=True)

    # horizontal error bars show the width of each mass bin
    xerr_low = df["mass_mid_GeV"] - df["mass_low_GeV"]
    xerr_high = df["mass_high_GeV"] - df["mass_mid_GeV"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(
        df["mass_mid_GeV"],
        df["sigma_fb"],
        yerr=df["error_fb"],
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#1f4e8c",
        ecolor="#1f4e8c",
        elinewidth=1,
        capsize=2,
        markersize=5,
        label=r"$gg \to H^* \to ZZ$ interference (MCFM proc. 129)",
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # symlog handles the sign flip near the Higgs pole while still
    # showing the ~1e-6 fb points near threshold and the ~4e-2 fb dip
    ax.set_yscale("symlog", linthresh=1e-5)

    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(r"$\sigma$ [fb]")
    ax.set_title(r"Higgs-continuum interference lineshape, $gg \to ZZ \to 4\ell$")

    # mark the on-shell Higgs mass for reference
    ax.axvline(125.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(115.0, 2.5e-4, r"$m_H$", color="gray", fontsize=13)

    # flag bins whose chi**2/iteration was not clean (suspect convergence)
    if "chisq_per_it" in df.columns:
        suspect = df[df["chisq_per_it"].fillna(0) > 5]
        if not suspect.empty:
            ax.scatter(
                suspect["mass_mid_GeV"],
                suspect["sigma_fb"],
                facecolors="none",
                edgecolors="red",
                s=140,
                linewidths=1.3,
                label="chi**2/it > 5 (use with caution)",
            )

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
            + ScaledTranslation(0, -4/72, fig.dpi_scale_trans)
        )

    out_pdf = csv_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(out_pdf)
    print(f"Saved plot to {out_pdf}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "interference_lineshape.csv"
    main(path)
