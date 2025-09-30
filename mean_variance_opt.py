from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.optimize import minimize
from matplotlib.patches import BoxStyle
import matplotlib as mpl
mpl.rcParams.update({
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
})


@dataclass
class FrontierResult:
    mu: float
    sigma: float
    weights: np.ndarray


def distinct_colors(n: int, banks: List[str] | None = None) -> List[tuple]:
    """
    Return n visually distinct RGBA colors without repeats.
    - Uses discrete tab palettes first (tab20, tab20b, tab20c).
    - Falls back to evenly spaced samples on a continuous map if needed.
    """
    if banks is None:
        banks = ["tab20", "tab20b", "tab20c"]

    pool: List[tuple] = []
    for name in banks:
        try:
            pool.extend(colormaps[name].colors)
        except Exception:
            # Not all backends have every map; skip quietly
            pass

    if n <= len(pool):
        return pool[:n]

    # Fallback: evenly spaced colors on HSV wheel (or another continuous cmap)
    cmap = colormaps.get("hsv")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


class EfficientFrontier:
    """
    End-to-end Markowitz frontier pipeline:
      - Load & clean prices
      - Resample (optional), compute returns, align time window
      - Annualize mean/cov
      - Min-variance and efficient frontier portfolios
      - Save CSV outputs and make plots
    """

    def __init__(self, infile: str, sheet: str, datecol: str, start: Optional[str] = None, 
                 resample: Optional[str] = None, allow_short: bool = False, lb: float = 0.0,
                 ub: float = 1.0, n_frontier: int = 50, outdir: str = "out", mapfile: Optional[str] = None, 
                 solver_maxiter: int = 1000, color_banks: List[str] | None = None,  # e.g., ["tab20","tab20b","tab20c"]
                 ):
        self.infile = infile
        self.sheet = sheet
        self.datecol = datecol
        self.start = start
        self.resample_rule = resample
        self.allow_short = allow_short
        self.lb = lb
        self.ub = ub
        self.n_frontier = n_frontier
        self.outdir = Path(outdir)
        self.mapfile = mapfile
        self.solver_maxiter = solver_maxiter
        self.color_banks = color_banks

        # Artifacts populated by fit()
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.assets: List[str] = []
        self.name_map: Dict[str, str] = {}
        self.asset_names: List[str] = []

        self.mean_ann: Optional[pd.Series] = None
        self.cov_ann: Optional[pd.DataFrame] = None
        self.frontier: List[FrontierResult] = []
        self.min_var: Optional[FrontierResult] = None

        # Derived/tabular views
        self.frontier_df: Optional[pd.DataFrame] = None
        self.weights_df: Optional[pd.DataFrame] = None
        self.weights_named_df: Optional[pd.DataFrame] = None
        self.min_var_weights: Optional[pd.Series] = None
        self.min_var_weights_named: Optional[pd.Series] = None

        self.outdir.mkdir(parents=True, exist_ok=True)

    # -------------
    # Data handling
    # -------------
    @staticmethod
    def _load_prices(path: str, sheet: str, datecol: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet)
        if datecol not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            datecol = "Date"
        df[datecol] = pd.to_datetime(df[datecol])
        df = df.set_index(datecol).sort_index()
        return df.select_dtypes(include=[np.number])

    @staticmethod
    def _resample_prices(px: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
        if not rule:
            return px
        return px.resample(rule).last()

    @staticmethod
    def _compute_returns(px: pd.DataFrame) -> pd.DataFrame:
        return px.pct_change().dropna(how="all")

    @staticmethod
    def _drop_sparse_assets(returns: pd.DataFrame, min_frac: float = 0.95) -> pd.DataFrame:
        counts = returns.notna().mean()
        keep = counts[counts >= min_frac].index
        return returns[keep]

    @staticmethod
    def _align_period(returns: pd.DataFrame, start: Optional[str] = None) -> pd.DataFrame:
        if start:
            returns = returns.loc[returns.index >= pd.to_datetime(start)]
        return returns.dropna(how="any")

    @staticmethod
    def _load_name_map(mapfile: Optional[str], assets: List[str]) -> Dict[str, str]:
        if not mapfile:
            return {t: t for t in assets}
        try:
            data = json.loads(Path(mapfile).read_text(encoding="utf-8"))
        except Exception:
            return {t: t for t in assets}
        return {t: data.get(t, t) for t in assets}

    # --------------------------
    # Annualization and frontier
    # --------------------------
    @staticmethod
    def _ann_stats(r: pd.DataFrame, freq: Optional[str] = "M") -> Tuple[pd.Series, pd.DataFrame]:
        if freq is None:
            if len(r) > 2:
                days = (r.index[-1] - r.index[0]).days
                periods_per_year = len(r) / (days / 365.25)
            else:
                periods_per_year = 252
        else:
            periods_per_year = {"D": 252, "W": 52, "M": 12}.get(freq[0], 12)
        mu = r.mean() * periods_per_year
        sigma = r.cov() * periods_per_year
        return mu, sigma

    def _solve_markowitz(self,mean: pd.Series, cov: pd.DataFrame,target_return: Optional[float],
                         allow_short: Optional[bool] = None,lb: Optional[float] = None,ub: Optional[float] = None,) -> FrontierResult:
        n = len(mean)
        mu = mean.values
        Sigma = cov.values

        def obj(w: np.ndarray) -> float:
            return float(w @ Sigma @ w)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target_return is not None:
            cons.append({"type": "eq", "fun": lambda w, mu=mu, tr=target_return: float(w @ mu - tr)})

        _allow_short = self.allow_short if allow_short is None else allow_short
        _lb = self.lb if lb is None else lb
        _ub = self.ub if ub is None else ub

        bounds = None if _allow_short else [(_lb, _ub)] * n
        w0 = np.ones(n) / n

        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": self.solver_maxiter},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        w = res.x
        port_mu = float(w @ mu)
        port_sig = float(np.sqrt(w @ Sigma @ w))
        return FrontierResult(mu=port_mu, sigma=port_sig, weights=w)

    def _min_variance(self, mean: pd.Series, cov: pd.DataFrame, **kw) -> FrontierResult:
        return self._solve_markowitz(mean, cov, target_return=None, **kw)

    def _frontier_curve(self,mean: pd.Series,cov: pd.DataFrame,n: int = 50,**kw,) -> List[FrontierResult]:
        _ = self._min_variance(mean, cov, **kw)  # validates feasibility
        n_assets = len(mean)
        mu = mean.values

        def neg_ret(w: np.ndarray) -> float:
            return float(-(w @ mu))

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        allow_short = kw.get("allow_short", self.allow_short)
        lb = kw.get("lb", self.lb)
        ub = kw.get("ub", self.ub)
        bounds = None if allow_short else [(lb, ub)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        res_max = minimize(neg_ret, w0, method="SLSQP", bounds=bounds, constraints=cons)
        res_min = minimize(lambda w: float(w @ mu), w0, method="SLSQP", bounds=bounds, constraints=cons)

        mu_max = float(res_max.x @ mu) if res_max.success else mean.max()
        mu_min = float(res_min.x @ mu) if res_min.success else mean.min()

        targets = np.linspace(mu_min, mu_max, n)
        results: List[FrontierResult] = []
        for tr in targets:
            try:
                fr = self._solve_markowitz(mean, cov, target_return=tr, **kw)
                results.append(fr)
            except Exception:
                continue
        return results

    # -------------------
    # Public API methods
    # -------------------
    def fit(self) -> "EfficientFrontier":
        px = self._load_prices(self.infile, self.sheet, self.datecol)
        if self.resample_rule:
            px = self._resample_prices(px, self.resample_rule)
        rets = self._compute_returns(px)
        rets = self._align_period(self._drop_sparse_assets(rets, min_frac=0.95), self.start)

        self.prices = px
        self.returns = rets
        self.assets = list(rets.columns)
        self.name_map = self._load_name_map(self.mapfile, self.assets)
        self.asset_names = [self.name_map.get(t, t) for t in self.assets]

        pd.Series(self.name_map).rename("name").to_csv(self.outdir / "ticker_name_map.csv")

        freq_for_ann = self.resample_rule if self.resample_rule else "M"
        self.mean_ann, self.cov_ann = self._ann_stats(rets, freq=freq_for_ann)

        self.frontier = self._frontier_curve(
            self.mean_ann, self.cov_ann, n=self.n_frontier,
            allow_short=self.allow_short, lb=self.lb, ub=self.ub
        )
        self.min_var = self._min_variance(
            self.mean_ann, self.cov_ann, allow_short=self.allow_short, lb=self.lb, ub=self.ub
        )

        self.frontier_df = pd.DataFrame(
            {"mu": [f.mu for f in self.frontier], "sigma": [f.sigma for f in self.frontier]}
        )
        self.weights_df = pd.DataFrame([f.weights for f in self.frontier], columns=self.assets)
        self.weights_df.index.name = "frontier_ix"

        self.min_var_weights = pd.Series(self.min_var.weights, index=self.assets, name="min_var_weights")

        self.weights_named_df = self.weights_df.copy()
        self.weights_named_df.columns = self.asset_names
        self.min_var_weights_named = self.min_var_weights.rename(index=self.name_map)
        self.min_var_weights_named.to_excel("min_variance_weights_named.xlsx")
        return self

    def save_outputs(self) -> None:
        if any(x is None for x in [self.frontier_df, self.weights_df, self.min_var_weights]):
            raise RuntimeError("Call .fit() before saving outputs.")

        self.frontier_df.to_csv(self.outdir / "efficient_frontier_stats.csv", index=False)
        self.weights_df.to_csv(self.outdir / "efficient_frontier_weights_tickers.csv")
        self.min_var_weights.to_csv(self.outdir / "min_variance_weights_tickers.csv")

        if self.weights_named_df is not None:
            self.weights_named_df.to_csv(self.outdir / "efficient_frontier_weights_named.csv")
        if self.min_var_weights_named is not None:
            self.min_var_weights_named.to_csv(self.outdir / "min_variance_weights_named.csv")

    # -------
    # Plots
    # -------
    def plot_frontier(self, show: bool = True, save: Optional[str] = None) -> None:
        if self.frontier_df is None:
            raise RuntimeError("Call .fit() before plotting.")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.frontier_df["sigma"], self.frontier_df["mu"], s=22)
        ticks_loc = ax.get_yticks().tolist()
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels(["{:.2%}".format(x) for x in ticks_loc])
        ticks_loc2 = ax.get_xticks().tolist()
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(["{:.2%}".format(x) for x in ticks_loc2])
        ax.grid(linestyle=":")
        ax.set_xlabel("Annualized Volatility (σ)")
        ax.set_ylabel("Annualized Return (μ)")
        title = "Efficient Frontier (Allow Short)" if self.allow_short else "Efficient Frontier (Long-only)"
        '''
        fig.suptitle((f'Efficient Frontier'),
                      fontsize=16, fontweight='bold', 
                      bbox=dict(facecolor='lightgrey',     # fill color
                                edgecolor='black',         # box border color
                                boxstyle='round,pad=0.6'   # rounded box with padding
                                ),
                     y=1.02  # push it up a bit above the top edge
                    )
        '''
        self.frontier_df["sigma"].to_excel('ERF.xlsx', index=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Frontier.eps', dpi=1000)
        if save:
            plt.savefig(self.outdir / save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        #plt.close()
        
    def plot_composition(self, show: bool = True, save: Optional[str] = None, 
                         colors: Optional[List[tuple]] = None) -> None:
        """
        Stackplot of frontier portfolio compositions by named assets.
        Ensures distinct colors (no repeats) by default via `distinct_colors`.
        """
        if self.weights_named_df is None:
            raise RuntimeError("Call .fit() before plotting.")

        n_assets = len(self.weights_named_df.columns)
        if colors is None:
            colors = distinct_colors(n_assets, banks=self.color_banks)

        fig, ax = plt.subplots(figsize=(8, 6))
        ticks_loc = ax.get_yticks().tolist()
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels(["{:.2%}".format(x) for x in ticks_loc])
        #ticks_loc2 = ax.get_xticks().tolist()
        #ax.set_xticks(ax.get_xticks().tolist())
        #ax.set_xticklabels(["{:.2%}".format(x) for x in ticks_loc2])
        ax.grid(linestyle=":")
        ax.set_xlabel("Annualized Volatility (σ)")
        ax.set_ylabel("Annualized Return (μ)")        
        #plt.set_title("Composition of Efficient Portfolios")
        plt.stackplot(range(len(self.weights_named_df)), self.weights_named_df.T.values, 
                      labels=self.weights_named_df.columns, colors=colors,
                     )
        leg = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, len(self.weights_named_df.columns)),
            fontsize=8,
            frameon=True,      # make box
            fancybox=True,     # rounded corners
        )       
        #leg.get_frame().set_boxstyle("round,pad=0.35,rounding_size=0.8")
        # Force a clearly visible box
        frame = leg.get_frame()
        frame.set_alpha(1.0)              # not transparent
        frame.set_facecolor("white")      # box fill
        frame.set_edgecolor("black")      # border color
        frame.set_linewidth(1.2)          # border width
        frame.set_boxstyle("round,pad=0.35,rounding_size=0.8")
        plt.xlabel("Frontier portfolios Mean Return (μ)")
        plt.ylabel("Stock Weight")
        #plt.title("Composition of Efficient Portfolios")
        '''
        fig.suptitle((f'Composition of Efficient Portfolios'),
                      fontsize=16, fontweight='bold', 
                      bbox=dict(facecolor='lightgrey',     # fill color
                                edgecolor='black',         # box border color
                                boxstyle='round,pad=0.6'   # rounded box with padding
                                ),
                      y= 1.02  # push it up a bit above the top edge
                    )
        '''    
        #plt.legend(loc="upper right", ncol=3, fontsize=8)
        plt.tight_layout()
        #fig.subplots_adjust(top=0.86)  # leave ~14% for the boxed title
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        #print("frame visible:", leg.get_frame().get_visible())
        #print("alpha:", leg.get_frame().get_alpha(), "edge:", leg.get_frame().get_edgecolor())

        plt.savefig('Composition.eps', dpi=1000)
        if save:
            plt.savefig(self.outdir / save, dpi=150)
        if show:
            plt.show()
        #plt.close()
        
    # ---------------
    # Convenience run
    # ---------------
    def run(self, make_plots: bool = True, save_plots: bool = False) -> "EfficientFrontier":
        self.fit()
        self.save_outputs()
        if make_plots:
            self.plot_frontier(show=True, save="efficient_frontier.png" if save_plots else None)
            self.plot_composition(show=True, save="efficient_frontier_composition_named.png" if save_plots else None)
        print(f"Saved outputs to: {self.outdir.resolve()}")
        return self


# ---------------------------
# Optional CLI-style harness
# ---------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Efficient Frontier pipeline")
    p.add_argument("--infile", required=True, help="Path to Excel prices file")
    p.add_argument("--sheet", required=True, help="Sheet name")
    p.add_argument("--datecol", required=True, help="Date column name (or first column is used)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD start date (optional)")
    p.add_argument("--resample", default=None, help="Pandas resample rule like 'M','W','Q' (optional)")
    p.add_argument("--allow-short", action="store_true", help="Allow shorting")
    p.add_argument("--lb", type=float, default=0.0, help="Lower bound for weights (if long-only)")
    p.add_argument("--ub", type=float, default=1.0, help="Upper bound for weights (if long-only)")
    p.add_argument("--n-frontier", type=int, default=50, help="Number of frontier points")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--mapfile", default=None, help="JSON file mapping tickers to names")
    p.add_argument("--save-plots", action="store_true", help="Save plots to files")
    args = p.parse_args()

    ef = EfficientFrontier(
        infile=args.infile,
        sheet=args.sheet,
        datecol=args.datecol,
        start=args.start,
        resample=args.resample,
        allow_short=args.allow_short,
        lb=args.lb,
        ub=args.ub,
        n_frontier=args.n_frontier,
        outdir=args.outdir,
        mapfile=args.mapfile,
    )
    ef.run(make_plots=True, save_plots=args.save_plots)
