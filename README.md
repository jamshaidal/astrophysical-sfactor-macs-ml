# Machine Learning for Astrophysical S-Factor & Maxwellian-Averaged Cross Sections (MACS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](#)

A reproducible scientific machine learning pipeline for modeling non-linear low-energy astrophysical S-factors and thermalized Maxwellian-Averaged Cross Sections (MACS) critical for stellar nucleosynthesis reaction networks.

---

## 🌌 Theoretical Background

In quiescent stellar burning environments, nuclear reactions take place at energies well below the Coulomb barrier (the Gamow window). Because the cross-section \(\sigma(E)\) drops exponentially due to quantum tunneling, it is parameterized via the **Astrophysical S-factor**:

$$S(E) = E \cdot \sigma(E) \cdot \exp(2\pi\eta)$$

where $\eta = \frac{Z_1 Z_2 e^2}{\hbar v}$ is the Sommerfeld parameter.

To calculate reaction rates in stellar plasmas, the cross section is integrated over a Maxwell-Boltzmann thermal velocity distribution to yield the **Maxwellian-Averaged Cross Section (MACS)**:

$$\langle \sigma v \rangle = \left(\frac{8}{\pi \mu (kT)^3}\right)^{1/2} \int_0^\infty \sigma(E) E \exp\left(-\frac{E}{kT}\right) dE$$

---

## 📊 Model Predictions & Validation

![Astrophysical S-Factor Predictions Plot](ML_S_Factor_Predictions_Publication.png)

*Figure: Machine learning surrogate model predictions across experimental nuclear reaction datasets, showing tight agreement with evaluated reaction benchmarks across the low-energy asymptotic limit.*

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/jamshaidal/astrophysical-sfactor-macs-ml.git
cd astrophysical-sfactor-macs-ml
pip install -r requirements.txt
```

### 2. Generate / Standardize Datasets
```bash
python generate_massive_ml_sfactor_dataset.py
```

### 3. Run Predictions & Plotting
```bash
python plot_ml_sfactor_results.py
```

---

## 📁 Repository Structure

```
.
├── generate_massive_ml_sfactor_dataset.py  # Dataset harmonization & feature engineering
├── plot_ml_sfactor_results.py              # ML evaluation & publication figure generation
├── ML_S_FACTOR_CLEAN_DATASET.csv           # Curated experimental benchmark dataset
├── ML_S_Factor_Predictions_Publication.png # High-resolution publication validation plot
└── requirements.txt                        # Computational dependencies
```

---

## 👨‍💻 Author & Inquiries

**Muhammad Jamshaid Ali**  
Computational Physics & Scientific ML Researcher  
- Email: [jamshaid8081@gmail.com](mailto:jamshaid8081@gmail.com)  
- LinkedIn: [linkedin.com/in/muhammad-jamshaid-ali-1687082a0](https://www.linkedin.com/in/muhammad-jamshaid-ali-1687082a0/)
