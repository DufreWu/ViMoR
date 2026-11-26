"""Common figure style: Times New Roman 16 pt; 2 subplots -> (8, 4), 3 subplots -> (15, 4).
Falls back to metric-compatible Times clones (Nimbus Roman / Liberation Serif) if TNR is not installed."""
import matplotlib

FIG1 = (8, 4)     # single panel (same size as a 2-panel figure)
FIG2 = (8, 4)
FIG3 = (15, 4)
FONTSIZE = 16


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # Times-like math
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "figure.titlesize": FONTSIZE,
        "pdf.fonttype": 42,                  # embed TrueType (editable text, accepted by IEEE PDF checks)
        "ps.fonttype": 42,
    })
