# IEEE Research Paper Compilation Instructions

This directory contains the complete, camera-ready scientific manuscript:
**"Trade-Offs Between Real-Time Inference Latency and Explanation Faithfulness in Credit Risk Scoring: A Reproducible Empirical Benchmark of TreeSHAP, KernelSHAP, FastSHAP, and LIME"**

## Files
- `main.tex`: Full IEEE double-column LaTeX manuscript with verified empirical results, formulas, tables, and figure references.
- `references.bib`: Verified BibTeX database of peer-reviewed literature, EU AI Act, and GDPR citations.
- `IEEEtran.cls`: Official IEEE conference class file (v1.8b).
- `figures/`: High-resolution publication figures generated programmatically from benchmark measurements.
- `tables/`: Measured result tables.

## How to Compile

### Option 1: Overleaf (Recommended for collaboration)
1. Zip the `paper/` folder:
   ```bash
   zip -r credit_xai_paper.zip paper/
   ```
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** > **Upload Project**, and select `credit_xai_paper.zip`.
3. Click **Recompile**. The paper compiles immediately with IEEEtran.

### Option 2: Local LaTeX Installation (MacTeX / TeX Live)
If MacTeX or TeX Live is installed:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Or using latexmk:
```bash
latexmk -pdf main.tex
```
