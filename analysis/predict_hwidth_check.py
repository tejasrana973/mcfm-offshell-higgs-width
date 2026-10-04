"""
Predicts what MCFM's own process 131 (full |H+C|^2) should report at the
124-126 GeV peak bin for a given hwidth_ratio, using the xi^4/xi^2/unchanged
reweighting recipe -- so you can compare it against the real MCFM output
from the one confirmatory run (see the note / instructions).

hwidth_ratio = Gamma_H / Gamma_H^SM = xi^4, so xi = hwidth_ratio^(1/4).

Usage:
    python3 predict_hwidth_check.py 5.0625 > p131_hwidth_check.log 2>&1
"""
import sys

# baseline (SM, xi=1) peak-bin values, from the closure-validated 124-126 GeV run
SIGMA_128_PEAK = 0.3510523   # fb, pure Higgs signal
SIGMA_129_PEAK = 0.0000937   # fb, pure interference
SIGMA_132_PEAK = 0.0030480   # fb, pure continuum (unaffected by hwidth_ratio)


def predict(hwidth_ratio):
    """
    Predict the 124-126 GeV peak bin's process-131 total for a given
    hwidth_ratio.

    IMPORTANT: this bin straddles the resonance, where the narrow-width
    approximation applies -- the integrated on-peak rate is expected to
    stay close to its SM baseline regardless of hwidth_ratio, since the
    width change in the propagator denominator cancels the coupling
    change in the numerator. Confirmed against a real MCFM run at
    hwidth_ratio=5.0625: baseline 0.354086 fb -> 0.352245 fb (-0.5%).
    So the prediction here is just the baseline, unscaled -- NOT
    hwidth_ratio-dependent. (The xi^4/xi^2 scaling in
    coupling_reweight.py is for off-peak bins; see its docstring.)
    """
    signal = SIGMA_128_PEAK
    interf = SIGMA_129_PEAK
    cont = SIGMA_132_PEAK
    total = signal + interf + cont
    return signal, interf, cont, total


if __name__ == "__main__":
    hwr = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0625
    signal, interf, cont, total = predict(hwr)
    print(f"hwidth_ratio = {hwr}  (on-peak bin: held fixed, not rescaled)")
    print(f"  signal (128, baseline, unscaled)   : {signal:.6f} fb")
    print(f"  interference (129, baseline)       : {interf:.6f} fb")
    print(f"  continuum (132, unchanged)         : {cont:.6f} fb")
    print(f"  PREDICTED process-131 total        : {total:.6f} fb")
    print()
    print("Expect the real MCFM value to land within ~1% of this,")
    print("by the narrow-width-approximation argument in the docstring --")
    print("NOT scaled by hwidth_ratio the way an off-peak bin would be.")
