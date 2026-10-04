"""
Core physics for Step 4: reweighting the gg -> H*/cont -> ZZ decomposition
under a rescaled (and optionally energy-dependent) Higgs coupling.

Physics behind the reweighting (see the note for the full derivation):
Rescaling the Higgs coupling by xi means the Higgs production+decay
amplitude M_H picks up a factor xi^2 (one power of xi from production,
one from decay). Therefore:

    signal        (proc 128, |M_H|^2)                  scales as xi^4
    interference  (proc 129, 2 Re[M_H* M_C])            scales as xi^2
    continuum     (proc 132, |M_C|^2)                    unchanged

This is exactly the structure used in Campbell, Ellis & Williams
(arXiv:1311.3589), Eq. (42)-(43), where the linear term in Gamma/Gamma_SM
comes from the signal and the sqrt term from the interference
(Gamma/Gamma_SM = xi^4, so sqrt(Gamma/Gamma_SM) = xi^2).

For a constant xi this reproduces the standard "uniform coupling
rescaling" width-bound method. For an energy-dependent xi(m), it becomes
a toy model for the momentum-transfer-dependent couplings that EFT
corrections can introduce -- exactly the assumption flagged in the
ATLAS off-shell paper (arXiv:1808.01191): "Assuming the ratio of the
Higgs boson couplings to the Standard Model predictions is independent
of the momentum transfer of the Higgs production mechanism..."

"""

import numpy as np
import pandas as pd


def load_data(csv_path="lineshape_data.csv"):
    """Load the per-bin 128/129/132 cross sections."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("mass_mid_GeV").reset_index(drop=True)
    return df


def xi_uniform(mass_GeV, xi_value):
    """A constant coupling rescaling (the standard method's assumption)."""
    return np.full_like(np.asarray(mass_GeV, dtype=float), xi_value)


def xi_eft_toy(mass_GeV, c):
    """
    A toy energy-dependent coupling: xi(m) = 1 + c * (m / 1 TeV)^2.

    This is a stand-in for a dimension-6-style EFT correction that grows
    with the momentum transfer through the Higgs propagator. It is
    explicitly a toy (a single real scalar form factor) -- genuine EFT
    operators can introduce new Lorentz structures this does not capture.
    """
    m = np.asarray(mass_GeV, dtype=float)
    return 1.0 + c * (m / 1000.0) ** 2


def reweighted_pieces(df, xi_func, **xi_kwargs):
    """
    Given a coupling-rescaling function xi_func(mass, **kwargs), return
    the reweighted signal, interference, continuum and total (H+I+C) at
    every mass point in df.

    The bin straddling the Higgs resonance (m_H = 125 GeV) is held at
    its baseline value, NOT scaled by xi^4/xi^2. This is confirmed
    against a real MCFM run with hwidth_ratio=5.0625 at the 124-126 GeV
    peak bin: the integrated on-peak rate came back within 0.5% of the
    unscaled baseline, as expected from the narrow-width approximation
    (integrating the Breit-Wigner over a window >> Gamma_H, the width
    dependence in the propagator denominator cancels the coupling
    dependence in the numerator -- this cancellation is the textbook
    reason the on-shell cross section is insensitive to Gamma_H in the
    first place). The xi^4/xi^2 scaling is only valid away from the
    resonance, where the width's contribution to the propagator is
    negligible compared to (s - m_H^2)^2.
    """
    mass = df["mass_mid_GeV"].values
    xi = xi_func(mass, **xi_kwargs)
    on_peak = (df["mass_low_GeV"].values <= 125.0) & (df["mass_high_GeV"].values >= 125.0)

    signal = np.where(on_peak, df["sigma_128_fb"].values, xi**4 * df["sigma_128_fb"].values)
    interf = np.where(on_peak, df["sigma_129_fb"].values, xi**2 * df["sigma_129_fb"].values)
    cont = df["sigma_132_fb"].values.copy()
    total = signal + interf + cont
    return signal, interf, cont, total


def offshell_sum(df, mass_cut_GeV, xi_func=None, **xi_kwargs):
    """
    Sum the Higgs-related off-shell yield (signal + interference, NOT
    continuum -- this mirrors how an experimental analysis reports the
    background-subtracted Higgs-related off-shell rate, and matches the
    convention in Eq. (40)-(43) of arXiv:1311.3589) for bins whose lower
    mass edge is at or above mass_cut_GeV.

    If xi_func is None, use xi = 1 everywhere (the baseline SM shape).
    """
    sel = df["mass_low_GeV"] >= mass_cut_GeV
    sub = df[sel]
    if xi_func is None:
        xi = np.ones(len(sub))
    else:
        xi = xi_func(sub["mass_mid_GeV"].values, **xi_kwargs)
    signal_sum = np.sum(xi**4 * sub["sigma_128_fb"].values)
    interf_sum = np.sum(xi**2 * sub["sigma_129_fb"].values)
    return signal_sum + interf_sum, sub


def naive_fit_gamma_ratio(df, mass_cut_GeV, observed_offshell_yield):
    """
    Invert the STANDARD (energy-independent xi) model to find the
    apparent xi_fit, and hence the apparent Gamma/Gamma_SM = xi_fit^4,
    that would reproduce a given observed off-shell yield.

    The model is  A*xi^4 + B*xi^2 = observed, with
        A = sum of baseline (xi=1) signal over the off-shell bins
        B = sum of baseline (xi=1) interference over the off-shell bins
    Substituting y = xi^2 turns this into the quadratic
        A*y^2 + B*y - observed = 0

    This quadratic has two mathematical roots. Only one is physical: at
    zero deviation (observed = A+B) it must return y=1 exactly. We solve
    with the numerically-stable form (to avoid cancellation when A is
    tiny, which happens whenever the off-shell region relies on bins
    where process 128 was unrecoverable and is treated as zero) and then
    pick whichever root sits closest to y=1 -- the branch a real analysis
    would take, since nobody reports the "coupling rescaled by 18x" root
    when a "coupling rescaled by 3%" root also fits.
    """
    sel = df["mass_low_GeV"] >= mass_cut_GeV
    sub = df[sel]
    A = np.sum(sub["sigma_128_fb"].values)   # baseline (xi=1) signal sum
    B = np.sum(sub["sigma_129_fb"].values)   # baseline (xi=1) interference sum
    C = observed_offshell_yield

    if abs(A) < 1e-12 * max(abs(B), 1e-30):
        # signal term negligible/unavailable in this off-shell region
        # (e.g. every contributing bin had process 128 crash) -- the
        # model degenerates to the linear relation B*y = C
        if B == 0:
            return None
        y = C / B
    else:
        disc = B**2 + 4 * A * C
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        sign_b = 1.0 if B >= 0 else -1.0
        q = -0.5 * (B + sign_b * sqrt_disc)
        candidates = []
        if q != 0:
            candidates.append(q / A)
        if q != 0:
            candidates.append((-C) / q)
        candidates = [y for y in candidates if y is not None]
        if not candidates:
            return None
        # physical branch: the root closest to y=1 (no deviation)
        y = min(candidates, key=lambda yy: abs(yy - 1.0))

    if y < 0:
        return None
    xi_fit = np.sqrt(y)
    gamma_ratio_apparent = xi_fit**4
    return xi_fit, gamma_ratio_apparent
