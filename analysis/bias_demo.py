"""
Step 4, capstone: show that a naive width extraction -- one that
assumes the coupling is energy-independent, exactly the assumption
named in arXiv:1808.01191 -- reports a FAKE width deviation when the
true physics has a momentum-transfer-dependent coupling (an EFT toy,
xi(m) = 1 + c*(m/1 TeV)^2), even though the true on-shell width never
changed at all.

Method (closed form, no fitting library needed):
  1. Pick a true c (the "hidden" EFT effect).
  2. Compute the true off-shell H+I yield using xi(m) at that c
     (offshell_sum with xi_eft_toy).
  3. Pretend that's what you measured. Invert the STANDARD model
     (constant xi, i.e. Gamma/Gamma_SM = xi^4) against that observed
     yield -- this is exactly what the Caola-Melnikov-style method does.
  4. Compare the apparent Gamma/Gamma_SM you'd report to the true value,
     which is 1 by construction (we never touched the actual width).

Usage:
    python3 bias_demo.py lineshape_data.csv > bias_demo.log 2>&1
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # Times-like math glyphs to match
plt.rcParams["font.size"] = 14          # base size; axis labels, ticks, legend all scale from this
# plt.rcParams["axes.titlesize"] = 14     # plot title
# plt.rcParams["axes.labelsize"] = 14     # x/y axis labels
# plt.rcParams["xtick.labelsize"] = 12    # x-axis tick numbers
# plt.rcParams["ytick.labelsize"] = 12    # y-axis tick numbers
plt.rcParams["legend.fontsize"] = 12    # legend text

from coupling_reweight import load_data, xi_eft_toy, offshell_sum, naive_fit_gamma_ratio


def run_table(df, mass_cut_GeV, c_values):
    rows = []
    for c in c_values:
        observed, _ = offshell_sum(df, mass_cut_GeV, xi_func=xi_eft_toy, c=c)
        result = naive_fit_gamma_ratio(df, mass_cut_GeV, observed)
        if result is None:
            rows.append((c, observed, None, None))
            continue
        xi_fit, gamma_ratio_apparent = result
        rows.append((c, observed, xi_fit, gamma_ratio_apparent))
    return rows


def print_table(rows, mass_cut_GeV):
    print(f"\nOff-shell region: m4l > {mass_cut_GeV} GeV")
    print(f"{'true c':>8} {'observed H+I (fb)':>20} {'xi_fit':>10} {'apparent Gamma/Gamma_SM':>26}")
    for c, obs, xi_fit, gr in rows:
        xi_s = f"{xi_fit:.4f}" if xi_fit is not None else "n/a"
        gr_s = f"{gr:.4f}" if gr is not None else "n/a"
        flag = "  <- truth is 1.0000, no real width change" if c != 0 and gr is not None else ""
        print(f"{c:>8.2f} {obs:>20.6e} {xi_s:>10} {gr_s:>26}{flag}")


def main(csv_path):
    df = load_data(csv_path)
    c_values = [-0.5, -0.2, 0.0, 0.2, 0.5]

    rows_130 = run_table(df, 130, c_values)
    rows_300 = run_table(df, 300, c_values)
    print_table(rows_130, 130)
    print_table(rows_300, 300)

    # capstone plot: apparent Gamma/Gamma_SM vs the true (hidden) c
    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, cut, color, marker in [
        (rows_130, 130, "#4c72b0", "o"),
        (rows_300, 300, "#c44e52", "s"),
    ]:
        cs = [r[0] for r in rows if r[3] is not None]
        grs = [r[3] for r in rows if r[3] is not None]
        ax.plot(cs, grs, marker=marker, color=color, label=rf"$m_{{4\ell}} > {cut}$ GeV cut")

    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--",
               label="truth (width never changed)")
    ax.set_xlabel(r"true EFT coefficient $c$ (hidden from the fit)")
    ax.set_ylabel(r"apparent $\Gamma_H/\Gamma_H^{\rm SM}$ from a naive constant-$\xi$ fit")
    ax.set_title("A naive width extraction is fooled by an energy-dependent coupling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_pdf = "bias_demo.pdf"
    fig.savefig(out_pdf)
    print(f"\nSaved {out_pdf}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lineshape_data.csv"
    main(path)
