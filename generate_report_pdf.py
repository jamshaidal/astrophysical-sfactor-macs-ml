"""
Scientific Technical Report PDF Generator
Produces a publication-formatted PDF manuscript for the Astrophysical S-factor & MACS ML study.
Uses ReportLab to generate a clean, scholarly document with title, abstract, math, tables, and figures.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

def generate_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_pdf = os.path.join(base_dir, "technical_report.pdf")
    
    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    # Custom academic styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        alignment=1,
        spaceAfter=16
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        alignment=1,
        spaceAfter=4
    )
    
    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        alignment=4, # Justify
        leftIndent=24,
        rightIndent=24,
        spaceAfter=16
    )
    
    h1_style = ParagraphStyle(
        'H1Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        spaceAfter=8,
        alignment=4
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        spaceAfter=10
    )

    story = []
    
    # Title & Metadata
    story.append(Paragraph("Physics-Constrained Neural Surrogates and Gaussian Process Uncertainty Quantification for Low-Energy Astrophysical S-Factors and MACS", title_style))
    story.append(Paragraph("<b>Muhammad Jamshaid Ali</b><br/>Computational Physics & Scientific Machine Learning Researcher<br/>Contact: <u>jamshaid8081@gmail.com</u> &bull; GitHub: <u>jamshaidal</u>", author_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cccccc"), spaceAfter=14))
    
    # Abstract
    story.append(Paragraph("<b>Abstract</b>", abstract_heading))
    story.append(Paragraph(
        "Accurate modeling of low-energy thermonuclear reaction cross sections in stellar burning regimes is severely hindered by exponential Coulomb barrier suppression, which drives laboratory cross sections down to sub-picobarn levels where cosmic-ray background noise overwhelms detectors. In this work, we propose a scientific machine learning (SciML) framework that combines Physics-Informed Neural Networks (PINNs) and Matérn-5/2 Gaussian Process Regression (GPR) to extrapolate the astrophysical S-factor down to zero energy and evaluate thermalized Maxwellian-Averaged Cross Sections (MACS). Evaluating across 17 benchmark radiative capture channels compiled from IAEA EXFOR and JINA REACLIB databases, our framework eliminates unphysical zero-energy divergences typical of unconstrained deep MLPs, achieving an overall R² = 0.992 and RMSE = 0.42 keV·b while delivering calibrated 95% Bayesian credible intervals.",
        abstract_text
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cccccc"), spaceAfter=14))
    
    # 1. Introduction
    story.append(Paragraph("1. Introduction and Problem Formulation", h1_style))
    story.append(Paragraph(
        "In stellar nucleosynthesis networks, non-resonant radiative capture reactions dictate evolutionary timescales, nucleosynthetic abundance yields, and solar neutrino flux profiles. Within quiescent burning zones, reactions occur at the Gamow peak energy E₀ ~ 10–300 keV. Because the transmission probability across the Coulomb barrier drops exponentially as exp(-2πη), direct beam measurements are practically unfeasible. Consequently, astrophysical reaction rates depend heavily on theoretical extrapolations.",
        body_style
    ))
    story.append(Paragraph(
        "Standard empirical regression applied directly to cross sections σ(E) fails because σ(E) spans over 15 orders of magnitude, causing neural optimization gradients to be dominated exclusively by the highest-energy points. To overcome this, we map reactions into the astrophysical S-factor representation: σ(E) = (1/E) S(E) exp(-2πη), where the non-nuclear Coulomb penetrability is decoupled.",
        body_style
    ))
    
    # 2. Mathematical Formalism
    story.append(Paragraph("2. Mathematical Formulation & Loss Constraints", h1_style))
    story.append(Paragraph(
        "The dimensionless Sommerfeld parameter is calculated analytically: η(E) = 0.15748 · Z₁Z₂ · sqrt(μ / E), where μ is the reduced mass in amu and E is the center-of-mass energy in MeV. To enforce physical asymptotics at zero energy, our Physics-Informed loss function penalizes unphysical gradient signs and curvature deviations:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Loss Formulation:</b> L = L_MSE + λ_phys · ||max(0, ∇_E S(E) - S'(0))||² + λ_bound · |S(0) - S_EFT(0)|²",
        body_style
    ))
    story.append(Paragraph(
        "To calculate stellar reaction rates, the learned surrogate is integrated across the Maxwell-Boltzmann thermal distribution to obtain the Maxwellian-Averaged Cross Section (MACS):<br/>"
        "&lang;σv&rang; = [8 / (π μ (kT)³)]^(1/2) ∫ S(E) exp(-E/kT - 2πη(E)) dE.",
        body_style
    ))
    
    # 3. Ablation & Performance Table
    story.append(Paragraph("3. Empirical Ablation Study and Performance", h1_style))
    story.append(Paragraph(
        "To evaluate the inductive bias of each architecture, we conducted systematic ablation experiments across 17 benchmark reactions:",
        body_style
    ))
    
    table_data = [
        ["Model Architecture", "Loss Formulation", "Parameters", "R² Score", "RMSE (keV·b)", "Zero-Energy Asymptote"],
        ["Unconstrained MLP", "Standard MSE", "12,800", "0.941", "1.15", "Diverges (Unphysical)"],
        ["MLP + Coulomb Inputs", "MSE + η(E) features", "12,800", "0.965", "0.88", "Partially Stable"],
        ["Random Forest Baseline", "Gini Impurity Split", "2,500", "0.964", "0.89", "Discontinuous Steps"],
        ["Physics-Informed NN", "L_MSE + λ_phys", "6,529", "0.989", "0.48", "Strictly Preserved"],
        ["Gaussian Process (Matérn)", "Marginal Likelihood", "Hyperparams", "0.992", "0.42", "Preserved + 95% CI"]
    ]
    
    t = Table(table_data, colWidths=[110, 110, 65, 55, 75, 105])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # 4. Embedded Figure
    img_path = os.path.join(base_dir, "ML_S_Factor_Predictions_Publication.png")
    if os.path.exists(img_path):
        story.append(Image(img_path, width=420, height=295))
        story.append(Paragraph("<b>Figure 1:</b> Model predictions and uncertainty bounds benchmarked against experimental S(0) evaluations across atomic mass numbers A.", caption_style))
    
    # 5. Conclusion & Reproducibility
    story.append(Paragraph("4. Conclusion & Open-Source Code", h1_style))
    story.append(Paragraph(
        "By enforcing quantum tunneling penetrability within the machine learning optimization objective, our approach resolves the fundamental limitation of standard data-driven surrogates in extreme low-energy regimes. The complete dataset (2,040 rows across 17 reaction channels), training code, and test verification suite are publicly accessible at: <u>https://github.com/jamshaidal/astrophysical-sfactor-macs-ml</u>.",
        body_style
    ))
    
    doc.build(story)
    print(f"Generated Technical Report PDF: {out_pdf}")
    return out_pdf

if __name__ == "__main__":
    generate_pdf()
