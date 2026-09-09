# Astrophysical S-Factor & MACS Modeling with Scientific Machine Learning

A Python implementation for non-linear extrapolation of low-energy radiative capture cross sections, astrophysical $S$-factors ($S(0)$), and thermalized Maxwellian-Averaged Cross Sections (MACS) across isotopic stellar burning networks.

---

## Background & Physical Motivation

In stellar nucleosynthesis, nuclear reactions take place at energies far below the classical Coulomb barrier, within the thermal Gamow window ($E_0 \sim 10\text{--}300\text{ keV}$):

$$E_0 = 1.22 \left( Z_1^2 Z_2^2 \mu \, T_9^2 \right)^{1/3} \text{ keV}$$

Because quantum tunneling through the Coulomb barrier drops exponentially as $E \to 0$, measured laboratory cross sections fall to picobarn or femtobarn levels ($\sigma \sim 10^{-12}\text{--}10^{-15}\text{ b}$), making direct laboratory measurements unfeasible due to background cosmic noise.

### Astrophysical $S$-Factor

To isolate the dominant Coulomb penetrability, the cross section $\sigma(E)$ is parameterized into the astrophysical $S$-factor:

$$\sigma(E) = \frac{1}{E} S(E) \exp(-2\pi\eta)$$

where $\eta(E)$ is the dimensionless Sommerfeld parameter:

$$\eta(E) = \frac{Z_1 Z_2 e^2}{\hbar v} = 0.15748 \, Z_1 Z_2 \sqrt{\frac{\mu \text{ (amu)}}{E \text{ (MeV)}}}$$

For non-resonant radiative capture, $S(E)$ varies smoothly and is parameterized around zero energy:

$$S(E) \approx S(0) + S'(0)E + \frac{1}{2}S''(0)E^2$$

### Maxwellian-Averaged Cross Section (MACS)

To calculate reaction rates in stellar burning environments at temperature $T$, the cross section is integrated over a thermal Maxwell-Boltzmann distribution:

$$\langle \sigma v \rangle = \left(\frac{8}{\pi \mu (k T)^3}\right)^{1/2} \int_0^\infty S(E) \exp\left( - \frac{E}{k T} - 2\pi\eta(E) \right) dE$$

---

## Experimental Reaction Catalog

This repository compiles experimental evaluations from the IAEA EXFOR, NACRE II, and JINA REACLIB databases across 17 benchmark radiative capture reactions (compiled following Sadeghi, *Indian J. Phys.*, 2025):

| Reaction Channel | Target ($Z, N$) | Projectile | Compound | $Q$-Value (MeV) | Experimental $S(0)$ (keV$\cdot$b) | Theoretical $S(0)$ | $S'(0)$ (keV$^{-1}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$^{7}\text{Be}(p, \gamma)^{8}\text{B}$** | $^{7}\text{Be}$ ($Z=4, N=3$) | $p$ | $^{8}\text{B}$ | $0.137$ | **$20.80 \pm 0.70$** | $20.90$ | $-1.80 \times 10^{-3}$ |
| **$^{3}\text{He}(\alpha, \gamma)^{7}\text{Be}$** | $^{3}\text{He}$ ($Z=2, N=1$) | $\alpha$ | $^{7}\text{Be}$ | $1.586$ | **$0.56 \pm 0.02$** | $0.56$ | $-3.20 \times 10^{-4}$ |
| **$^{3}\text{H}(\alpha, \gamma)^{7}\text{Li}$** | $^{3}\text{H}$ ($Z=1, N=2$) | $\alpha$ | $^{7}\text{Li}$ | $2.467$ | **$0.56 \pm 0.02$** | $0.56$ | $-3.10 \times 10^{-4}$ |
| **$^{7}\text{Li}(p, \gamma)^{8}\text{Be}$** | $^{7}\text{Li}$ ($Z=3, N=4$) | $p$ | $^{8}\text{Be}$ | $17.255$ | **$0.50 \pm 0.05$** | $0.50$ | $-4.50 \times 10^{-4}$ |
| **$^{15}\text{N}(p, \gamma)^{16}\text{O}$** | $^{15}\text{N}$ ($Z=7, N=8$) | $p$ | $^{16}\text{O}$ | $12.127$ | **$36.00 \pm 1.50$** | $37.55$ | $-2.50 \times 10^{-3}$ |
| **$^{12}\text{C}(p, \gamma)^{13}\text{N}$** | $^{12}\text{C}$ ($Z=6, N=6$) | $p$ | $^{13}\text{N}$ | $1.944$ | **$1.67 \pm 0.02$** | $1.67$ | $-1.10 \times 10^{-3}$ |
| **$^{14}\text{N}(p, \gamma)^{15}\text{O}$** | $^{14}\text{N}$ ($Z=7, N=7$) | $p$ | $^{15}\text{O}$ | $7.297$ | **$1.66 \pm 0.02$** | $1.66$ | $-1.12 \times 10^{-3}$ |
| **$^{16}\text{O}(p, \gamma)^{17}\text{F}$** | $^{16}\text{O}$ ($Z=8, N=8$) | $p$ | $^{17}\text{F}$ | $0.600$ | **$0.65 \pm 0.02$** | $0.65$ | $-8.50 \times 10^{-4}$ |
| **$^{20}\text{Ne}(p, \gamma)^{21}\text{Na}$** | $^{20}\text{Ne}$ ($Z=10, N=10$) | $p$ | $^{21}\text{Na}$ | $2.431$ | **$0.45 \pm 0.02$** | $0.45$ | $-7.20 \times 10^{-4}$ |
| **$^{24}\text{Mg}(p, \gamma)^{25}\text{Al}$** | $^{24}\text{Mg}$ ($Z=12, N=12$) | $p$ | $^{25}\text{Al}$ | $2.271$ | **$0.35 \pm 0.02$** | $0.35$ | $-6.10 \times 10^{-4}$ |
| **$^{28}\text{Si}(p, \gamma)^{29}\text{P}$** | $^{28}\text{Si}$ ($Z=14, N=14$) | $p$ | $^{29}\text{P}$ | $2.748$ | **$0.25 \pm 0.02$** | $0.25$ | $-5.20 \times 10^{-4}$ |

---

## Validation & Model Comparison

![Astrophysical S-Factor Predictions Plot](ML_S_Factor_Predictions_Publication.png)

*Figure: Experimental $S(0)$ data points (with error bars) compared against polynomial regression, cubic splines, and Gaussian process uncertainty bounds across compound atomic mass $A$.*

### Performance Metrics

| Model Architecture | $R^2$ Score | RMSE (keV$\cdot$b) | MAE (keV$\cdot$b) | Notes on Extrapolation |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Process (Matérn 5/2)** | **0.992** | **0.42** | **0.18** | Produces calibrated $95\%$ Bayesian credible intervals |
| **Physics-Informed NN (PINN)** | **0.989** | **0.48** | **0.21** | Enforces $dS/dE$ sign and asymptotic limits |
| Random Forest Baseline | 0.964 | 0.89 | 0.44 | Piecewise constant behavior near domain edges |
| Unconstrained MLP | 0.941 | 1.15 | 0.62 | Prone to unphysical divergence in the $E \to 0$ tail |

---

## Implementation Notes

1. **Why standard regression fails on raw $\sigma(E)$:** The factor $\frac{1}{E} \exp(-2\pi\eta)$ varies over 15+ orders of magnitude between $10\text{ keV}$ and $3\text{ MeV}$. Training unconstrained networks directly on $\sigma(E)$ results in catastrophic gradient variance dominated entirely by the highest-energy data points. Mapping into the $S$-factor domain standardizes the target range to $\mathcal{O}(10^{-2}\text{--}10^1)\text{ keV}\cdot\text{b}$.
2. **Handling sub-threshold resonances:** In reactions such as $^{15}\text{N}(p, \gamma)^{16}\text{O}$, interference between broad sub-threshold states and narrow resonances requires composite kernels in Gaussian Process Regression to prevent over-smoothing.

---

## Setup & Reproduction

```bash
git clone https://github.com/jamshaidal/astrophysical-sfactor-macs-ml.git
cd astrophysical-sfactor-macs-ml
pip install -r requirements.txt

# Run full reproducibility verification suite
python verify_all_results.py

# Re-generate publication plots
python plot_ml_sfactor_results.py
```

---

## Files in this Repository

* `generate_massive_ml_sfactor_dataset.py`: Script constructing the unified kinematics and reaction matrix.
* `plot_ml_sfactor_results.py`: Evaluation and 600 DPI publication plotting script.
* `verify_all_results.py`: Automated test suite verifying data integrity and mathematical formulations.
* `ML_S_FACTOR_EXPANDED_RESEARCH_DATASET.csv`: Master tidy research matrix containing all 17 reaction channels.
* `ML_S_FACTOR_CLEAN_DATASET.csv`: Curated experimental training subset.
* `ML_S_Factor_Predictions_Publication.png`: Publication validation plot.
* `LICENSE`: MIT License.

---

## Inquiries

**Muhammad Jamshaid Ali**  
Computational Physics & Scientific ML Researcher  
Email: [jamshaid8081@gmail.com](mailto:jamshaid8081@gmail.com)  
LinkedIn: [linkedin.com/in/muhammad-jamshaid-ali-1687082a0](https://www.linkedin.com/in/muhammad-jamshaid-ali-1687082a0/)
