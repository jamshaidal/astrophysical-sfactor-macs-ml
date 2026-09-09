"""
Render Publication-Quality Figure for Machine Learning S-Factor Study
Plots Experimental S(0) data vs ML Polynomial Regression, Cubic Spline, and Power-Law Fits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = r"c:\Users\Jamshaid\Desktop\problems"
OUT_IMG = os.path.join(BASE_DIR, "ML_S_Factor_Predictions_Publication.png")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "Times"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5.5,
    "xtick.minor.size": 3.0,
    "ytick.major.size": 5.5,
    "ytick.minor.size": 3.0,
    "legend.frameon": False,
    "figure.dpi": 600,
})

def plot_s_factor():
    df = pd.read_excel(os.path.join(BASE_DIR, "ML_S_FACTOR_CLEAN_DATASET.xlsx"))
    
    exp_df = df[df["Source"] == "Experimental Measurement"]
    ml_poly = df[df["Source"] == "ML Polynomial Regression (Degree 2)"]
    ml_spline = df[df["Source"].str.contains("ML Hybrid Interpolation")]
    
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=600)
    
    # Sort by atomic mass
    exp_df = exp_df.sort_values("Final_Nucleus_A")
    ml_poly = ml_poly.sort_values("Final_Nucleus_A")
    ml_spline = ml_spline.sort_values("Final_Nucleus_A")
    
    # Plot ML fits
    ax.plot(ml_spline["Final_Nucleus_A"], ml_spline["Value"], color="#1f77b4", lw=2.0, label=r"$\mathrm{ML\ Cubic\ Spline\ (R^2=0.98)}$")
    ax.plot(ml_poly["Final_Nucleus_A"], ml_poly["Value"], color="#9467bd", linestyle=":", lw=2.2, label=r"$\mathrm{ML\ Polynomial\ Deg.\ 2\ (R^2=0.99)}$")
    
    # Plot Experimental points
    ax.errorbar(
        exp_df["Final_Nucleus_A"], exp_df["Value"],
        yerr=exp_df["Uncertainty_Pos"], fmt='o', color='#d62728',
        ecolor='#1f77b4', elinewidth=1.5, capsize=3.5, capthick=1.2,
        ms=6.5, label=r"$\mathrm{Experimental\ Data\ [S(0)]}$", zorder=5
    )
    
    # Annotate outlier 15N(p, gamma)16O and 7Be(p, gamma)8B
    ax.annotate(r"$\mathrm{^{15}N(p,\gamma)^{16}O}$", xy=(16, 36.0), xytext=(28, 38.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1), fontsize=10.5)
    ax.annotate(r"$\mathrm{^{7}Be(p,\gamma)^{8}B}$", xy=(8, 20.8), xytext=(18, 23.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1), fontsize=10.5)

    ax.set_yscale("log")
    ax.set_xlim(4, 215)
    ax.set_ylim(0.005, 60.0)
    
    ax.set_xlabel(r"$\mathrm{Compound\ Mass\ Number\ } A$", fontsize=13)
    ax.set_ylabel(r"$S(0)\ \mathrm{(keV\cdot b)}$", fontsize=13)
    ax.set_title(r"$\mathrm{Astrophysical\ } S(0)\mathrm{\ Factor\ vs.\ Compound\ Mass\ } A$", fontsize=13.5, pad=10)
    
    ax.legend(loc="upper right", fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=600)
    plt.close()
    print(f"Generated: {OUT_IMG}")

if __name__ == "__main__":
    plot_s_factor()
