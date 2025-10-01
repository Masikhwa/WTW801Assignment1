
# WTW 801 Assignment 1 — Portfolio Optimisation on JSE (Part A & Part B)

> **Status:** Work-in-progress • **Focus:** Markowitz Efficient Frontier (Part A) & Exercises 8.6–8.7 (Part B) from the book: Optimization Mtehos in Finance.  JSE data  from yahoo finance is used
> **Author:** Muofhe Masikhwa • **Course:** WTW 801

This repository contains reproducible code and report assets for **WTW 801 Assignment 1**. It builds efficient frontiers on JSE assets (Part A), and solves **Exercises 8.6 & 8.7** (Part B) with additional variants such as beta-band and cap‑group constraints. Where relevant, it also scaffolds **Black–Litterman (Part B.1)** and includes LaTeX for **KKT‑based derivations (Part B.2)**.

---

## 📁 Repository Structure

```
.
├─ data/
│  ├─ raw/
│  │  └─ jse_prices_adjclose_new.xlsx    # Provided adjusted close prices (Date column + tickers)
│  ├─ external/
│  │  ├─ jse_top40_pre1998.json          # Ticker ➜ Company name map (optional)
│  │  └─ cap_groups.csv                  # Cap-group mapping (e.g., Large/Mid/Small)
│  └─ processed/                         # Generated monthly returns & intermediate files
├─ notebooks/
│  ├─ 01_partA_efficient_frontier.ipynb  # Part A analysis (long-only / various options)
│  └─ 02_partB_exercises_8_6_8_7.ipynb   # Part B walkthrough + diagnostics
├─ src/
│  ├─ __init__.py
│  ├─ efficient_frontier.py              # EfficientFrontier class & helpers
│  ├─ part2_exercises.py                 # CLI for Exercises 8.6 & 8.7
│  ├─ bl_model.py                        # (Optional) Black–Litterman scaffolding (Part B.1)
│  └─ utils/
│     ├─ io_utils.py                     # Excel/CSV read, resampling, mapping
│     └─ plotting.py                     # Plot styles, distinct colours, legends
├─ reports/
│  ├─ partA/                             # Figures & tables exported by Part A
│  └─ partB/                             # Figures & tables exported by Part B
├─ tex/
│  ├─ partB2_kkt_derivation.tex          # LaTeX: full mathematical derivation (KKT)
│  └─ references.bib                     # (Optional) bibliography
├─ requirements.txt                      # Minimal Python dependencies
├─ pyproject.toml                        # (Optional) project metadata if packaging
├─ LICENSE
└─ README.md
```

