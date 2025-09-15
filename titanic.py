#!/usr/bin/env python3
"""
Titanic EDA: Single-file HTML Report Generator
----------------------------------------------
- Loads `train.csv` (Kaggle Titanic format)
- Creates key EDA visuals with matplotlib
- Embeds images as base64 into one HTML file
- Writes: titanic_eda.html

Usage:
    python titanic_eda.py --csv path/to/train.csv --out titanic_eda.html
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fig_to_base64() -> str:
    """Save current matplotlib figure to base64-encoded PNG and close it."""
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_images(df: pd.DataFrame) -> dict:
    """Create plots and return a dict of base64 images."""
    images = {}

    # 1) Survival Count
    surv_counts = df["Survived"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["No (0)", "Yes (1)"], [surv_counts.get(0, 0), surv_counts.get(1, 0)])
    plt.title("Survival Count")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    images["survival_count"] = fig_to_base64()

    # 2) Survival by Sex (grouped bars)
    ct = pd.crosstab(df["Sex"], df["Survived"]).reindex(["male", "female"])
    x = np.arange(len(ct.index))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, ct[0].values, width, label="Not Survived (0)")
    plt.bar(x + width / 2, ct[1].values, width, label="Survived (1)")
    plt.xticks(x, ct.index)
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.title("Survival by Sex")
    plt.legend()
    images["survival_by_sex"] = fig_to_base64()

    # 3) Survival by Pclass (grouped bars)
    ct2 = pd.crosstab(df["Pclass"], df["Survived"]).sort_index()
    x = np.arange(len(ct2.index))
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, ct2[0].values, width, label="Not Survived (0)")
    plt.bar(x + width / 2, ct2[1].values, width, label="Survived (1)")
    plt.xticks(x, ct2.index.astype(str))
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.title("Survival by Passenger Class")
    plt.legend()
    images["survival_by_pclass"] = fig_to_base64()

    # 4) Age Distribution
    plt.figure(figsize=(7, 4))
    plt.hist(df["Age"].dropna().values, bins=30, edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    images["age_dist"] = fig_to_base64()

    # 5) Fare Distribution
    plt.figure(figsize=(7, 4))
    plt.hist(df["Fare"].dropna().values, bins=30, edgecolor="black")
    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Frequency")
    images["fare_dist"] = fig_to_base64()

    # 6) Correlation Heatmap (numeric only, matplotlib)
    num = df.select_dtypes(include=[np.number]).copy()
    labels = num.columns.tolist()
    corr = num.corr(numeric_only=True).values
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title("Correlation Heatmap (Numeric Features)")
    images["corr_heatmap"] = fig_to_base64()

    return images


def dict_to_table(d: dict) -> str:
    rows = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items()])
    return f"""<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"""


def generate_html(df: pd.DataFrame, images: dict) -> str:
    total = len(df)
    survival_rate = df["Survived"].mean()
    missing = df.isna().mean().mul(100).round(2).to_dict()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Titanic EDA Report</title>
<style>
  :root {{ --bg:#0b1020; --fg:#eef2ff; --muted:#b9c0d4; --card:#121836; --accent:#8ea5ff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--fg); }}
  header {{ padding: 24px; border-bottom: 1px solid #1e254a; background: linear-gradient(180deg, #0e1538, #0b1020); position: sticky; top:0; z-index: 1; }}
  h1 {{ margin: 0 0 6px; font-size: 28px; }}
  .subtitle {{ margin:0; color: var(--muted); }}
  main {{ max-width: 1100px; margin: 24px auto; padding: 0 16px 48px; }}
  section {{ background: var(--card); border: 1px solid #1e254a; border-radius: 16px; padding: 18px; margin-bottom: 18px; }}
  h2 {{ margin-top: 0; font-size: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  figure {{ margin: 0; background:#0e1330; border:1px solid #1e254a; border-radius: 12px; padding: 12px; }}
  figcaption {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #1e254a; padding: 8px 10px; text-align: left; }}
  th {{ color:#c7d0ff; }}
  .pill {{ display:inline-block; padding:4px 10px; border:1px solid #2a3370; border-radius: 999px; color:#cbd5ff; background:#121a45; }}
  .kpi {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .kpi .card {{ background:#0f1538; border:1px solid #1f2655; border-radius:12px; padding:10px 12px; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 14px; }}
  .footer {{ color:#95a1d8; text-align:center; margin-top: 16px; }}
</style>
</head>
<body>
<header>
  <h1>Titanic EDA Report</h1>
  <p class="subtitle">Single-file HTML • auto-generated</p>
</header>
<main>

<section>
  <h2>Summary</h2>
  <div class="kpi">
    <div class="card"><div class="pill">Rows</div><div><strong>{total}</strong></div></div>
    <div class="card"><div class="pill">Columns</div><div><strong>{df.shape[1]}</strong></div></div>
    <div class="card"><div class="pill">Survival Rate</div><div><strong>{survival_rate:.2%}</strong></div></div>
  </div>
  <p class="muted small">Key insight: Survival is strongly related to sex and class. Age has missing values; Cabin is mostly missing.</p>
</section>

<section>
  <h2>Missing Data (%)</h2>
  {dict_to_table(missing)}
</section>

<section>
  <h2>Descriptive Statistics</h2>
  <div class="grid">
    <figure>
      <img alt="Survival Count" src="data:image/png;base64,{images['survival_count']}" style="width:100%;height:auto;" />
      <figcaption>More passengers did not survive than survived.</figcaption>
    </figure>
    <figure>
      <img alt="Survival by Sex" src="data:image/png;base64,{images['survival_by_sex']}" style="width:100%;height:auto;" />
      <figcaption>Females had a much higher survival count than males.</figcaption>
    </figure>
    <figure>
      <img alt="Survival by Pclass" src="data:image/png;base64,{images['survival_by_pclass']}" style="width:100%;height:auto;" />
      <figcaption>1st class shows many more survivors than 2nd and 3rd.</figcaption>
    </figure>
    <figure>
      <img alt="Age Distribution" src="data:image/png;base64,{images['age_dist']}" style="width:100%;height:auto;" />
      <figcaption>Most passengers were between 20–40 years old.</figcaption>
    </figure>
    <figure>
      <img alt="Fare Distribution" src="data:image/png;base64,{images['fare_dist']}" style="width:100%;height:auto;" />
      <figcaption>Fare is right-skewed with a few very high values.</figcaption>
    </figure>
    <figure>
      <img alt="Correlation Heatmap" src="data:image/png;base64,{images['corr_heatmap']}" style="width:100%;height:auto;" />
      <figcaption>Survival correlates positively with Fare and negatively with Pclass.</figcaption>
    </figure>
  </div>
</section>

<section>
  <h2>Quick Facts</h2>
  <ul>
    <li>Women survived far more often than men.</li>
    <li>1st class passengers had significantly better outcomes.</li>
    <li>Age missing: {missing.get('Age',0)}% • Cabin missing: {missing.get('Cabin',0)}%</li>
  </ul>
</section>

<p class="footer small">Generated by titanic_eda.py</p>
</main>
</body>
</html>"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="train.csv", help="Path to Titanic train.csv")
    ap.add_argument("--out", type=str, default="titanic_eda.html", help="Output HTML file")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    images = build_images(df)
    html = generate_html(df, images)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
