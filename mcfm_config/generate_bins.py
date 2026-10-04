#!/usr/bin/env python3
"""
Generate MCFM input files for a binned M(ZZ) lineshape scan.

Run this from MCFM-10.3/Bin, where input_p128.ini, input_p129.ini,
input_p131.ini and input_p132.ini already exist (the wide-window,
already-validated versions with cuts/scale/PDF/precision set).

For each process, writes one input_p<proc>_bin<NN>.ini per mass bin,
changing only m3456min, m3456max, and runstring.
"""

import re
from pathlib import Path

# GeV edges: finer near the 125 GeV resonance, coarser in the smooth tail.
# The 124-126 bin is intentionally left out -- you already ran and
# validated it as input_p<proc>_pk.ini, no need to duplicate that job.
BIN_EDGES = [100, 115, 120, 124, 126, 130, 140, 170, 220, 300, 450, 650, 1000]

PROCESSES = [128, 129, 131, 132]
BASE_DIR = Path(".")


def set_key(text: str, key: str, value) -> str:
    """Replace the value on a `key = ...` line, uncommenting it if needed.
    Raises if the key doesn't appear exactly once, so a typo or a missing
    key fails loudly instead of silently writing a bad file."""
    pattern = re.compile(rf"^(\s*)#?\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
    new_line = rf"\g<1>{key} = {value}"
    new_text, n = pattern.subn(new_line, text, count=1)
    if n != 1:
        raise ValueError(f"expected exactly one '{key}' line, found {n}")
    return new_text


def main():
    for proc in PROCESSES:
        base_file = BASE_DIR / f"input_p{proc}.ini"
        if not base_file.exists():
            raise FileNotFoundError(
                f"{base_file} not found -- run this from MCFM-10.3/Bin"
            )
        base_text = base_file.read_text()

        for i in range(len(BIN_EDGES) - 1):
            lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
            # skip the bin that's already been run as the "_pk" peak window
            if (lo, hi) == (124, 126):
                continue
            runstring = f"p{proc}_bin{i:02d}"
            text = base_text
            text = set_key(text, "m3456min", lo)
            text = set_key(text, "m3456max", hi)
            text = set_key(text, "runstring", runstring)
            out_file = BASE_DIR / f"input_{runstring}.ini"
            out_file.write_text(text)
            print(f"wrote {out_file.name:30s} [{lo:4d}, {hi:4d}) GeV")


if __name__ == "__main__":
    main()