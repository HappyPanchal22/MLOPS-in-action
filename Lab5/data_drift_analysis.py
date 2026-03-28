# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ks_2samp, chi2_contingency

from ucimlrepo import fetch_ucirepo

# New Evidently API (v0.6+)
from evidently import Report
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset

print("=" * 65)
print("  Evidently AI – Data Drift on UCI Bank Marketing Dataset")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────
# 2. DATASET LOADING & EXPLORATION
# ──────────────────────────────────────────────────────────────────────
print("\n📦 Fetching Bank Marketing dataset (UCI ID: 222)...")
bank = fetch_ucirepo(id=222)

# Combine features and targets into one DataFrame
df = pd.concat([bank.data.features, bank.data.targets], axis=1)
print(f"✅ Loaded {df.shape[0]} records with {df.shape[1]} columns.\n")

print("── First 5 Rows ──")
print(df.head())
print(f"\n── Shape: {df.shape} ──")
print(f"\n── Column Types ──")
print(df.dtypes)
print(f"\n── Missing Values ──")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().any().any() else "No missing values found.")

# ──────────────────────────────────────────────────────────────────────
# 3. FEATURE SCHEMA DEFINITION
# ──────────────────────────────────────────────────────────────────────
numerical_features = ["age", "balance", "duration", "campaign", "pdays", "previous"]
categorical_features = ["job", "marital", "education", "default", "housing", "loan",
                        "contact", "poutcome", "y"]

# Verify all columns exist
all_features = numerical_features + categorical_features
missing_cols = [c for c in all_features if c not in df.columns]
if missing_cols:
    print(f"⚠️  Columns not found in dataset: {missing_cols}")
else:
    print(f"\n✅ Schema validated: {len(numerical_features)} numerical, "
          f"{len(categorical_features)} categorical features selected.")

# ──────────────────────────────────────────────────────────────────────
# 4. DATA SPLITTING – TEMPORAL SPLIT BY CONTACT MONTH
# ──────────────────────────────────────────────────────────────────────
print("\n── Splitting Data by Contact Month ──")
print(f"Month distribution:\n{df['month'].value_counts().sort_index()}\n")

# Reference = May (peak campaign), Production = November (off-season)
ref_data = df[df["month"] == "may"].reset_index(drop=True)
prod_data = df[df["month"] == "nov"].reset_index(drop=True)

# Drop the 'month' and 'day' columns (not useful for drift on features)
drop_cols = ["month", "day"]
ref_data = ref_data.drop(columns=[c for c in drop_cols if c in ref_data.columns])
prod_data = prod_data.drop(columns=[c for c in drop_cols if c in prod_data.columns])

print(f"📌 Reference dataset (May):      {ref_data.shape[0]} records")
print(f"📌 Production dataset (November): {prod_data.shape[0]} records")

# ──────────────────────────────────────────────────────────────────────
# 5. EVIDENTLY DATA DRIFT REPORT (New API v0.6+)
# ──────────────────────────────────────────────────────────────────────
print("\n⚙️  Running Evidently DataDriftPreset report...")

# Define data definition with column types
data_definition = DataDefinition(
    numerical_columns=numerical_features,
    categorical_columns=categorical_features
)

# Create Evidently Dataset objects
ref_dataset = Dataset.from_pandas(
    ref_data,
    data_definition=data_definition
)
prod_dataset = Dataset.from_pandas(
    prod_data,
    data_definition=data_definition
)

# Create and run the report
# In Evidently 0.7+, report.run() returns a result object
report = Report([DataDriftPreset()])
my_eval = report.run(prod_dataset, ref_dataset)

# Save interactive HTML dashboard
report_path = "bank_marketing_drift_report.html"
my_eval.save_html(report_path)
print(f"✅ Evidently HTML report saved to: {report_path}")

# Extract drift results programmatically
try:
    report_dict = my_eval.dict()
    print("\n📊 Evidently report generated successfully.")
    print(f"   Open '{report_path}' in your browser for the interactive dashboard.")
    print(f"   Report dict keys: {list(report_dict.keys())}\n")
except Exception as e:
    print(f"\n📊 Evidently HTML report generated successfully.")
    print(f"   Open '{report_path}' in your browser for the interactive dashboard.")
    print(f"   (Note: dict export encountered: {e})\n")

# ──────────────────────────────────────────────────────────────────────
# 6. MANUAL DRIFT DETECTION (KS-test & Chi-square)
# ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  📐 MANUAL DRIFT DETECTION (KS-test & Chi-Square)")
print("=" * 65)

drift_records = []

# Numerical features: Kolmogorov-Smirnov test
for feat in numerical_features:
    if feat in ref_data.columns and feat in prod_data.columns:
        ref_vals = ref_data[feat].dropna()
        prod_vals = prod_data[feat].dropna()
        ks_stat, ks_pval = ks_2samp(ref_vals, prod_vals)
        drifted = ks_pval < 0.05
        drift_records.append({
            "Feature": feat,
            "Type": "numerical",
            "Stat Test": "Kolmogorov-Smirnov",
            "Test Statistic": round(ks_stat, 4),
            "P-Value": f"{ks_pval:.2e}",
            "Drift Detected": "✅" if drifted else "❌"
        })

# Categorical features: Chi-Square test
for feat in categorical_features:
    if feat in ref_data.columns and feat in prod_data.columns:
        try:
            # Build contingency table
            all_cats = sorted(set(ref_data[feat].dropna().unique()) |
                              set(prod_data[feat].dropna().unique()))
            ref_counts = ref_data[feat].value_counts().reindex(all_cats, fill_value=0)
            prod_counts = prod_data[feat].value_counts().reindex(all_cats, fill_value=0)
            contingency = pd.DataFrame({"ref": ref_counts, "prod": prod_counts})
            chi2, p_val, dof, _ = chi2_contingency(contingency.T)
            drifted = p_val < 0.05
            drift_records.append({
                "Feature": feat,
                "Type": "categorical",
                "Stat Test": "Chi-Square",
                "Test Statistic": round(chi2, 4),
                "P-Value": f"{p_val:.2e}",
                "Drift Detected": "✅" if drifted else "❌"
            })
        except Exception as e:
            print(f"⚠️  Skipping {feat}: {e}")

drift_df = pd.DataFrame(drift_records)
drift_df = drift_df.sort_values("Test Statistic", ascending=False).reset_index(drop=True)
print(f"\n{drift_df.to_string(index=False)}")

n_columns = len(drift_df)
n_drifted = len(drift_df[drift_df["Drift Detected"] == "✅"])
drift_share = n_drifted / n_columns if n_columns > 0 else 0

print(f"\n{'=' * 50}")
print(f"  📊 DRIFT SUMMARY")
print(f"{'=' * 50}")
print(f"  Total columns analyzed : {n_columns}")
print(f"  Drifted columns        : {n_drifted}")
print(f"  Drift share            : {drift_share:.2%}")
print(f"  Dataset drift detected : {'✅ YES' if drift_share > 0.5 else '❌ NO'}")
print(f"  Threshold              : 0.5")
print(f"{'=' * 50}")

# Export to CSV
drift_csv_path = "drift_summary_table.csv"
drift_df.to_csv(drift_csv_path, index=False)
print(f"\n✅ Drift summary table exported to: {drift_csv_path}")

# ──────────────────────────────────────────────────────────────────────
# 7. ADDITIONAL STATISTICAL TESTS – PSI & JENSEN-SHANNON DIVERGENCE
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  📐 ADDITIONAL STATISTICAL TESTS (PSI & Jensen-Shannon)")
print("=" * 65)


def compute_psi(reference, production, bins=10):
    """Compute Population Stability Index (PSI) for a numerical feature."""
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        bins + 1
    )
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    eps = 1e-4
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    prod_pct = (prod_counts + eps) / (prod_counts.sum() + eps * bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi


def compute_js_divergence(reference, production, bins=10):
    """Compute Jensen-Shannon Divergence for a numerical feature."""
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        bins + 1
    )
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    eps = 1e-10
    ref_prob = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    prod_prob = (prod_counts + eps) / (prod_counts.sum() + eps * bins)

    m = 0.5 * (ref_prob + prod_prob)
    jsd = 0.5 * entropy(ref_prob, m, base=2) + 0.5 * entropy(prod_prob, m, base=2)
    return jsd


extra_stats = []
for feat in numerical_features:
    if feat in ref_data.columns and feat in prod_data.columns:
        ref_vals = ref_data[feat].dropna()
        prod_vals = prod_data[feat].dropna()

        psi_val = compute_psi(ref_vals, prod_vals)
        jsd_val = compute_js_divergence(ref_vals, prod_vals)
        ks_stat, ks_pval = ks_2samp(ref_vals, prod_vals)

        extra_stats.append({
            "Feature": feat,
            "PSI": round(psi_val, 4),
            "PSI Verdict": "🔴 Major" if psi_val > 0.25 else ("🟡 Moderate" if psi_val > 0.1 else "🟢 Stable"),
            "JS Divergence": round(jsd_val, 4),
            "KS Statistic": round(ks_stat, 4),
            "KS P-Value": f"{ks_pval:.2e}"
        })

extra_stats_df = pd.DataFrame(extra_stats)
print(f"\n{extra_stats_df.to_string(index=False)}")

print("""
PSI Interpretation Guide:
  🟢 PSI < 0.10  → No significant drift (distributions stable)
  🟡 0.10 ≤ PSI < 0.25 → Moderate drift (investigate further)
  🔴 PSI ≥ 0.25  → Major drift (retrain model recommended)
""")

# ──────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE RANKING FOR DRIFTED FEATURES
# ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  🏆 FEATURE IMPORTANCE RANKING (by Drift Severity)")
print("=" * 65)

drifted_features = drift_df[drift_df["Drift Detected"] == "✅"].copy()

if not drifted_features.empty:
    # Merge PSI scores for numerical features
    psi_map = {row["Feature"]: row["PSI"] for row in extra_stats}
    drifted_features["PSI"] = drifted_features["Feature"].map(psi_map).fillna(0)

    # Composite score: normalize test_statistic and PSI, then average
    ts_col = drifted_features["Test Statistic"].astype(float)
    psi_col = drifted_features["PSI"].astype(float)

    ts_norm = (ts_col - ts_col.min()) / (ts_col.max() - ts_col.min() + 1e-10)
    psi_norm = (psi_col - psi_col.min()) / (psi_col.max() - psi_col.min() + 1e-10)

    drifted_features["Composite Score"] = ((ts_norm + psi_norm) / 2).round(4)
    drifted_features = drifted_features.sort_values("Composite Score", ascending=False)

    print("\nRank | Feature                | Test Stat   | PSI    | Composite")
    print("-" * 70)
    for rank, (_, row) in enumerate(drifted_features.iterrows(), 1):
        print(f"  {rank:<2} | {row['Feature']:<22} | {row['Test Statistic']:<11} | "
              f"{row['PSI']:<6} | {row['Composite Score']}")
else:
    print("No drifted features detected.")

# ──────────────────────────────────────────────────────────────────────
# 9. VISUALIZATIONS – DISTRIBUTION COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────────
print("\n\n📊 Generating distribution comparison plots...")

# --- 9a. Overlapping Histograms for Numerical Features ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Distribution Comparison: Reference (May) vs Production (Nov)",
             fontsize=16, fontweight="bold", y=1.02)

for idx, feat in enumerate(numerical_features):
    ax = axes[idx // 3][idx % 3]
    ax.hist(ref_data[feat].dropna(), bins=30, alpha=0.6, label="Ref (May)",
            color="#2196F3", edgecolor="white", density=True)
    ax.hist(prod_data[feat].dropna(), bins=30, alpha=0.6, label="Prod (Nov)",
            color="#FF5722", edgecolor="white", density=True)
    ax.set_title(feat, fontsize=13, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("numerical_histograms.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: numerical_histograms.png")

# --- 9b. Box Plots for Numerical Features ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Box Plot Comparison: Reference (May) vs Production (Nov)",
             fontsize=16, fontweight="bold", y=1.02)

for idx, feat in enumerate(numerical_features):
    ax = axes[idx // 3][idx % 3]
    combined = pd.DataFrame({
        "Value": pd.concat([ref_data[feat], prod_data[feat]], ignore_index=True),
        "Dataset": (["Reference (May)"] * len(ref_data[feat])) +
                   (["Production (Nov)"] * len(prod_data[feat]))
    })
    sns.boxplot(data=combined, x="Dataset", y="Value", ax=ax,
                palette={"Reference (May)": "#2196F3", "Production (Nov)": "#FF5722"})
    ax.set_title(feat, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("numerical_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: numerical_boxplots.png")

# --- 9c. Stacked Bar Charts for Top Categorical Features ---
top_cat_features = ["job", "marital", "education", "poutcome"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Categorical Feature Distribution: Reference vs Production",
             fontsize=16, fontweight="bold", y=1.02)

for idx, feat in enumerate(top_cat_features):
    ax = axes[idx // 2][idx % 2]

    ref_counts = ref_data[feat].value_counts(normalize=True).sort_index()
    prod_counts = prod_data[feat].value_counts(normalize=True).sort_index()

    all_cats = sorted(set(ref_counts.index) | set(prod_counts.index))
    ref_aligned = ref_counts.reindex(all_cats, fill_value=0)
    prod_aligned = prod_counts.reindex(all_cats, fill_value=0)

    x = np.arange(len(all_cats))
    width = 0.35
    ax.bar(x - width / 2, ref_aligned.values, width, label="Ref (May)",
           color="#2196F3", edgecolor="white")
    ax.bar(x + width / 2, prod_aligned.values, width, label="Prod (Nov)",
           color="#FF5722", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title(feat, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("categorical_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: categorical_distributions.png")

# --- 9d. PSI Heatmap Bar Chart ---
if extra_stats:
    fig, ax = plt.subplots(figsize=(10, 5))
    psi_df = extra_stats_df.sort_values("PSI", ascending=True)

    colors = []
    for v in psi_df["PSI"]:
        if v >= 0.25:
            colors.append("#F44336")
        elif v >= 0.1:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    ax.barh(psi_df["Feature"], psi_df["PSI"], color=colors, edgecolor="white")
    ax.axvline(x=0.1, color="orange", linestyle="--", label="Moderate Threshold (0.10)")
    ax.axvline(x=0.25, color="red", linestyle="--", label="Major Threshold (0.25)")
    ax.set_xlabel("PSI Value", fontsize=12)
    ax.set_title("Population Stability Index (PSI) by Feature",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("psi_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: psi_bar_chart.png")

# ──────────────────────────────────────────────────────────────────────
# 10. FINAL SUMMARY & KEY INSIGHTS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  💡 KEY INSIGHTS & CONCLUSIONS")
print("=" * 65)
print(f"""
1. DATASET OVERVIEW:
   The UCI Bank Marketing dataset (ID: 222) contains {df.shape[0]} records
   from a Portuguese bank's phone-based marketing campaign. The data was
   split temporally — May (peak season) as reference, November (off-season)
   as production — to simulate real-world deployment drift.

2. DRIFT DETECTION:
   Manual statistical tests detected drift in {n_drifted}/{n_columns}
   features ({drift_share:.1%}). {'This exceeds' if drift_share > 0.5 else 'This is below'}
   the 50% threshold for dataset-level drift.

3. PSI ANALYSIS:
   Population Stability Index provides a complementary view.
   Features with PSI > 0.25 warrant immediate attention
   and likely require model retraining or feature engineering adjustments.

4. JENSEN-SHANNON DIVERGENCE:
   JSD values close to 0 indicate similar distributions; values approaching
   1 indicate maximum divergence. This metric is symmetric and bounded,
   making it more interpretable than raw KL divergence.

5. RECOMMENDATIONS:
   - Monitor high-drift features (especially 'duration', 'pdays', 'previous')
     in production pipelines.
   - Consider time-aware feature engineering or periodic retraining.
   - Integrate Evidently reports into CI/CD for automated drift alerts.
   - Use PSI thresholds as automated retrain triggers in MLOps workflows.
""")

print("🏁 Script complete. Check generated files:")
print("   • bank_marketing_drift_report.html  (Evidently interactive dashboard)")
print("   • drift_summary_table.csv           (exportable drift results)")
print("   • numerical_histograms.png          (distribution histograms)")
print("   • numerical_boxplots.png            (box plot comparisons)")
print("   • categorical_distributions.png     (categorical bar charts)")
print("   • psi_bar_chart.png                 (PSI severity chart)")