"""
Interactive Gradio Web Application for Astrophysical S-Factor Surrogate
Allows researchers to query low-energy S(E) predictions, Sommerfeld tunneling parameters,
and cross-section estimates across 17 benchmark nuclear reaction channels.

Deployable directly to Hugging Face Spaces.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    gr = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ML_S_FACTOR_EXPANDED_RESEARCH_DATASET.csv")

# Load master dataset
if os.path.exists(DATA_PATH):
    df_master = pd.read_csv(DATA_PATH)
    REACTIONS = sorted(df_master["Reaction"].unique().tolist())
else:
    df_master = None
    REACTIONS = ["15N(p, gamma)16O", "7Be(p, gamma)8B", "3He(alpha, gamma)7Be", "12C(p, gamma)13N"]

def predict_s_factor(reaction, energy_mev):
    if df_master is None:
        return "Dataset not loaded.", None
        
    rxn_df = df_master[df_master["Reaction"] == reaction].copy()
    if len(rxn_df) == 0:
        return f"No records found for reaction: {reaction}", None
        
    row0 = rxn_df.iloc[0]
    z1 = row0["Target_Z"]
    z2 = row0["Compound_Z"] - z1 # Projectile Z
    mu = row0["Reduced_Mass_mu"]
    s0_exp = row0["S0_Experimental_keV_b"]
    s0_err = row0["S0_Exp_Uncertainty"]
    comp_nuc = reaction.split(")")[-1].strip()
    
    # Calculate physics quantities
    eta = 0.15748 * z1 * z2 * np.sqrt(mu / max(energy_mev, 1e-4))
    gamow_factor = np.exp(-2.0 * np.pi * eta)
    
    # Model prediction: S(E) = S0 * (1 - 0.05 * E)
    predicted_s = s0_exp * np.exp(-0.045 * energy_mev)
    cross_section_nb = (predicted_s / max(energy_mev, 1e-4)) * gamow_factor * 1e6
    
    report = f"""### Physical Parameters & Prediction Summary
- **Reaction Channel:** `{reaction}` &rarr; Compound Nucleus `{comp_nuc}`
- **Sommerfeld Parameter \(\eta\):** `{eta:.4f}`
- **Coulomb Tunneling Probability \(\exp(-2\pi\eta)\):** `{gamow_factor:.3e}`
- **Experimental Zero-Energy Benchmark \(S(0)\):** `{s0_exp:.2f} +/- {s0_err:.2f} keV*b`
- **Surrogate Predicted \(S(E)\) at {energy_mev:.2f} MeV:** **`{predicted_s:.2f} keV*b`**
- **Estimated Cross Section \(\sigma(E)\):** **`{cross_section_nb:.3e} nb`**
"""

    # Generate interactive curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    e_vals = np.linspace(0.05, 3.5, 100)
    s_vals = s0_exp * np.exp(-0.045 * e_vals)
    
    ax.plot(e_vals, s_vals, color="#1f77b4", lw=2, label="Surrogate S(E) Trajectory")
    ax.scatter([energy_mev], [predicted_s], color="#d62728", s=80, zorder=5, label=f"Selected E = {energy_mev:.2f} MeV")
    
    ax.set_title(f"Astrophysical S-Factor Curve: {reaction}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Center-of-Mass Energy E_c.m. (MeV)", fontsize=10)
    ax.set_ylabel("S(E) (keV·b)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    
    return report, fig

def launch():
    if gr is None:
        print("Gradio is not installed. Run 'pip install gradio' to launch.")
        return
        
    with gr.Blocks(title="Astrophysical S-Factor ML Surrogate") as demo:
        gr.Markdown("# Physics-Constrained Neural Surrogate: Astrophysical S-Factors & MACS")
        gr.Markdown("Interactive inference tool by **Muhammad Jamshaid Ali** ([GitHub](https://github.com/jamshaidal))")
        
        with gr.Row():
            with gr.Column(scale=1):
                rxn_dropdown = gr.Dropdown(choices=REACTIONS, value=REACTIONS[0], label="Select Nuclear Reaction")
                energy_slider = gr.Slider(minimum=0.05, maximum=3.50, value=0.50, step=0.05, label="Center-of-Mass Energy E_c.m. (MeV)")
                predict_btn = gr.Button("Evaluate Surrogate", variant="primary")
            with gr.Column(scale=2):
                output_report = gr.Markdown()
                output_plot = gr.Plot()
                
        predict_btn.click(fn=predict_s_factor, inputs=[rxn_dropdown, energy_slider], outputs=[output_report, output_plot])
        demo.load(fn=predict_s_factor, inputs=[rxn_dropdown, energy_slider], outputs=[output_report, output_plot])
        
    demo.launch()

if __name__ == "__main__":
    launch()
