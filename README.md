
# WTW 801 Assignment 1 — Portfolio Optimisation on JSE (Part A & Part B)

> **Status:** Work-in-progress • **Focus:** Markowitz Efficient Frontier (Part A) & Exercises 8.6–8.7 (Part B) using JSE data  
> **Author:** Ndivhuwo Mphephu • **Course:** WTW 801

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

> If you don’t see some of these files yet, they will be created by the notebooks/CLI during the first run.

---

## 🔧 Quick Start

### 1) Create environment
```bash
# Option A: conda
conda create -n wtw801 python=3.11 -y
conda activate wtw801
pip install -r requirements.txt

# Option B: venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Put the data in place
Place your **adjusted close** Excel file here:
```
data/raw/jse_prices_adjclose_new.xlsx
```
Expected layout:
- A `Date` column (YYYY‑MM‑DD or similar)
- One numeric column per ticker (e.g., `NPN.JO`, `AGL.JO`, …)

Optional add‑ons:
- `data/external/jse_top40_pre1998.json` — map `{ "NPN.JO": "Naspers Ltd", ... }`
- `data/external/cap_groups.csv` — two columns: `ticker,cap_group`

### 3) Run Part A (Efficient Frontier)
Use the notebook `notebooks/01_partA_efficient_frontier.ipynb` **or** the class directly from a Python script.
Minimal example:
```python
from src.efficient_frontier import EfficientFrontier
import pandas as pd

px = pd.read_excel("data/raw/jse_prices_adjclose_new.xlsx", sheet_name=0)
px["Date"] = pd.to_datetime(px["Date"])
px = px.set_index("Date").sort_index()

ef = EfficientFrontier.from_prices(px, freq="M", start="1998-01-01")
res = ef.solve_frontier(n_points=25, long_only=True)  # returns list of (mu, sigma, weights)
ef.plot_composition(res, name_map_json="data/external/jse_top40_pre1998.json",
                    legend_loc="lower center", legend_bbox_to_anchor=(0.5, -0.3))
```

Outputs (by default) are saved under `reports/partA/` with tidy legends and non‑repeating colours.

### 4) Run Part B — Exercises 8.6 & 8.7 (CLI)
The CLI computes **monthly returns** since a start date, and then solves 8.6 (beta‑band) and 8.7 (perturbed‑means average portfolio).

```bash
python -m src.part2_exercises \
  --infile data/raw/jse_prices_adjclose_new.xlsx \
  --sheet 0 --datecol Date \
  --start 1998-01-01 \
  --mapfile data/external/jse_top40_pre1998.json \
  --capmap data/external/cap_groups.csv \
  --outdir reports/partB
```

Key flags:
- `--start`: first date for return computation
- `--mapfile`: optional ticker→name mapping for prettier plots/tables
- `--capmap`: optional cap‑group constraints for 8.6(ii)
- `--outdir`: where to write figures and CSVs

### 5) (Optional) Part B.1 — Black–Litterman with an added asset
A simple scaffold is provided in `src/bl_model.py`:
- Compute equilibrium returns from market cap weights
- Encode investor views (absolute and/or relative)
- Blend with prior via τ and Ω choices
- Re‑optimise portfolio and compare to Markowitz baseline

See `notebooks/02_partB_exercises_8_6_8_7.ipynb` for examples.

### 6) Part B.2 — KKT Derivation (LaTeX)
Open the LaTeX file:
```
tex/partB2_kkt_derivation.tex
```
Compile with:
```bash
pdflatex tex/partB2_kkt_derivation.tex
```

---

## 📊 Notes on Plotting (Distinct Colours & Legends)

- Distinct qualitative palettes are combined (`tab20`, `tab20b`, `tab20c`, etc.), with HSV fallback to avoid repeats.
- To keep the **title clear** and off the plot edge:
  ```python
  fig.suptitle("Composition of Efficient Portfolios",
               fontsize=16, fontweight="bold",
               bbox=dict(facecolor="lightgrey", edgecolor="black", boxstyle="round,pad=0.6"),
               y=1.02)
  ```
- To place the legend **outside at the bottom** while showing **all assets**:
  ```python
  ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35),
            ncols=4, frameon=True, borderaxespad=0.5)
  ```

---

## 🧪 Reproducibility

- Set the random seed where optimization/perturbations are used.
- All generated tables/figures are timestamped and written under `reports/`.
- For deterministic results, pin versions in `requirements.txt`.

---

## 🧰 Requirements

Minimal dependencies (install via `pip install -r requirements.txt`):
```
numpy
pandas
matplotlib
scipy
openpyxl
```

Optional (used in some notebooks):
```
riskfolio-lib
PyPortfolioOpt
```

---

## 🚦 Troubleshooting

- **Dates not found / wrong column name**  
  Ensure the Excel has a `Date` column; override with `--datecol` if needed.
- **Yahoo session errors**  
  This project reads from a local Excel file, avoiding Yahoo API issues entirely.
- **`reset_index(names=...)` TypeError**  
  Use `reset_index().rename(columns={0: "dividend"})` for older pandas.

---

## 📄 License & Academic Integrity

This repository is distributed for learning and reproducible research. If your course policy restricts code sharing, keep the repo private until submission and **cite all external sources** used in your report.

---

## 🙌 Acknowledgements

- Lecturers and TAs of **WTW 801**
- Open‑source libraries: NumPy, pandas, SciPy, Matplotlib, RiskFolio‑Lib, PyPortfolioOpt

---

## 🔗 Citation

If you reference this work in a report:
```
Mphephu, N. (2025). WTW 801 Assignment 1 — Portfolio Optimisation on JSE (Part A & Part B). GitHub repository.
```

---

## 🧭 Roadmap

- [ ] Finalise Black–Litterman (view specification & Ω calibration)
- [ ] Add unit tests for `EfficientFrontier`
- [ ] Export publication‑quality figures (TikZ/PGF)
- [ ] Expand cap‑group and factor‑exposure constraints
