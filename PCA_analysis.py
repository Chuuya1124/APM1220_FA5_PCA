# APM1220 FA5 - PCA on World Happiness Report 2021 (script version of the notebook)

# Display shim for non-notebook use

try:
    display
except NameError:
    display = print


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.options.display.float_format = "{:.3f}".format
pd.set_option("display.width", 200)

import os
FILE = "world-happiness-report-2021.csv"
if not os.path.exists(FILE):
    raise FileNotFoundError('Put the CSV next to this script')

raw = pd.read_csv(FILE)
print("Full dataset shape:", raw.shape)
raw.head()

# Map the seven required variables (robust to small column-name differences)
wanted = {
    "Ladder Score": "Ladder score",
    "Logged GDP per capita": "Logged GDP per capita",
    "Social Support": "Social support",
    "Healthy Life Expectancy": "Healthy life expectancy",
    "Freedom to Make Life Choices": "Freedom to make life choices",
    "Generosity": "Generosity",
    "Perceptions of Corruption": "Perceptions of corruption",
}
lower = {c.lower().strip(): c for c in raw.columns}
cols = [lower[v.lower()] for v in wanted.values()]

country_col = lower["country name"]
data = raw[cols].copy()
data.columns = list(wanted.keys())          # short, readable names
countries = raw[country_col]                # kept ONLY for labelling plots, NOT used in PCA

n_obs, n_vars = data.shape
print(f"Number of observations (countries): {n_obs}")
print(f"Number of variables: {n_vars}")
print("Variables:", list(data.columns))
print("Missing values per variable:\n", data.isna().sum())

desc = pd.DataFrame({"Mean": data.mean(), "Std. Dev.": data.std(ddof=1)})
desc

corr = data.corr()
display(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title("Correlation Matrix of the Seven Well-Being Variables")
plt.tight_layout(); plt.show()

# Strongest pairs (by absolute correlation)
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().rename("r").reset_index())
pairs.columns = ["Variable 1", "Variable 2", "r"]
pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
print("Top 5 strongest pairs:")
display(pairs.head(5))

# Standardize: z = (x - mean) / sd
scaler = StandardScaler()
Z = scaler.fit_transform(data)
Z_df = pd.DataFrame(Z, columns=data.columns)
print("Means of standardized data (≈0):", np.round(Z_df.mean().values, 3))
print("Std devs of standardized data (≈1):", np.round(Z_df.std(ddof=0).values, 3))

# PCA = eigen-decomposition of the correlation matrix
R = np.corrcoef(Z, rowvar=False)                    # equals data.corr()
eigvals, eigvecs = np.linalg.eigh(R)
order = np.argsort(eigvals)[::-1]                   # sort descending
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# Sign convention: make the largest |coefficient| of each PC positive (signs are arbitrary)
for j in range(eigvecs.shape[1]):
    if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
        eigvecs[:, j] *= -1

pc_names = [f"PC{i+1}" for i in range(n_vars)]
print("Check: sum of eigenvalues =", round(eigvals.sum(), 3), "(equals number of variables =", n_vars, ")")

# Cross-check with scikit-learn
skl = PCA().fit(Z)
print("sklearn variance ratios match eigen-decomposition:",
      np.allclose(skl.explained_variance_ratio_, eigvals / eigvals.sum()))

prop = eigvals / eigvals.sum()          # lambda_k / p
cum = np.cumsum(prop)
summary = pd.DataFrame({"Eigenvalue": eigvals,
                        "Proportion of Variance": prop,
                        "Cumulative Proportion": cum}, index=pc_names)
summary

eig_df = pd.DataFrame(eigvecs, index=data.columns, columns=pc_names)
print("All eigenvectors (columns = PCs):")
display(eig_df)

# Decide how many components to retain (Kaiser, set again in Part C)
k_kaiser = int((eigvals > 1).sum())
print(f"\nComponents with eigenvalue > 1: {k_kaiser}")
print("Eigenvectors of the retained components:")
display(eig_df.iloc[:, :k_kaiser])

pc1 = eig_df["PC1"]
pc1_rank = pc1.reindex(pc1.abs().sort_values(ascending=False).index)
display(pc1_rank.to_frame("PC1 coefficient").assign(**{"|Coefficient|": pc1_rank.abs()}))
print("Largest |coefficients| in PC1:", ", ".join(pc1_rank.index[:3]))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

ax[0].plot(range(1, n_vars+1), eigvals, "o-", lw=2)
ax[0].axhline(1, color="red", ls="--", label="Kaiser line (eigenvalue = 1)")
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Eigenvalue")
ax[0].set_title("Scree Plot"); ax[0].set_xticks(range(1, n_vars+1)); ax[0].legend()
for i, v in enumerate(eigvals): ax[0].annotate(f"{v:.3f}", (i+1, v), textcoords="offset points", xytext=(6, 6))

ax[1].bar(range(1, n_vars+1), prop, alpha=.6, label="Individual")
ax[1].plot(range(1, n_vars+1), cum, "o-", color="darkorange", label="Cumulative")
ax[1].axhline(0.80, color="gray", ls=":", label="80% reference")
ax[1].set_xlabel("Principal component"); ax[1].set_ylabel("Proportion of variance")
ax[1].set_title("Variance Explained"); ax[1].set_xticks(range(1, n_vars+1)); ax[1].legend()
plt.tight_layout(); plt.show()

# Drops between successive eigenvalues help locate the elbow
drops = pd.DataFrame({"Drop from previous PC": -np.diff(eigvals)},
                     index=[f"PC{i}→PC{i+1}" for i in range(1, n_vars)])
display(drops)

kaiser = summary[summary["Eigenvalue"] > 1]
print(f"Kaiser criterion retains {len(kaiser)} component(s):")
display(kaiser)

k = k_kaiser                      # change this if your scree plot / interpretation suggests otherwise
print(f"Retained components: {k}")
print(f"Cumulative variance explained by retained components: {cum[k-1]:.3f} ({cum[k-1]*100:.1f}%)")
print(f"Variance reduced from {n_vars} variables to {k} components.")

# Loadings = eigenvector * sqrt(eigenvalue) = correlation between each variable and each PC
loadings = pd.DataFrame(eigvecs * np.sqrt(eigvals), index=data.columns, columns=pc_names)
print("Loadings (variable–component correlations):")
display(loadings.iloc[:, :max(k, 2)])

plt.figure(figsize=(7, 5))
sns.heatmap(loadings.iloc[:, :max(k, 2)], annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Loadings of Retained Components"); plt.tight_layout(); plt.show()

# Auto-generated reading of PC1 and PC2 (threshold |coef| >= 0.35 treated as notable)
for pc in ["PC1", "PC2"]:
    s = eig_df[pc]
    pos = s[s >= 0.35].sort_values(ascending=False)
    neg = s[s <= -0.35].sort_values()
    print(f"\n{pc}  (eigenvalue {eigvals[int(pc[2])-1]:.3f}, {prop[int(pc[2])-1]*100:.1f}% of variance)")
    print("  Notable positive:", ", ".join(f"{i} ({v:.3f})" for i, v in pos.items()) or "none")
    print("  Notable negative:", ", ".join(f"{i} ({v:.3f})" for i, v in neg.items()) or "none")

scores = Z @ eigvecs[:, :2]
score_df = pd.DataFrame(scores, columns=["PC1", "PC2"]); score_df["Country"] = countries.values

fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(score_df["PC1"], score_df["PC2"], s=18, alpha=.5)
# label extreme countries
extreme = pd.concat([score_df.nlargest(5, "PC1"), score_df.nsmallest(5, "PC1"),
                     score_df.nlargest(4, "PC2"), score_df.nsmallest(4, "PC2")]).drop_duplicates("Country")
for _, r in extreme.iterrows(): ax.annotate(r["Country"], (r["PC1"], r["PC2"]), fontsize=8)

scale = np.abs(scores).max() * 0.85
for var in data.columns:
    x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
    ax.arrow(0, 0, x, y, color="red", head_width=0.08, alpha=.8)
    ax.text(x * 1.08, y * 1.08, var, color="red", fontsize=9, ha="center")
ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
ax.set_xlabel(f"PC1 ({prop[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({prop[1]*100:.1f}%)")
ax.set_title("PCA Biplot: Countries and Variable Loadings"); plt.tight_layout(); plt.show()

print("Top 5 countries on PC1:", ", ".join(score_df.nlargest(5, "PC1")["Country"]))
print("Bottom 5 countries on PC1:", ", ".join(score_df.nsmallest(5, "PC1")["Country"]))

# Communality = share of each variable's variance reproduced by the k retained components
communality = (loadings.iloc[:, :k] ** 2).sum(axis=1).to_frame(f"Communality (first {k} PCs)")
display(communality)
print(f"Average communality: {communality.iloc[:,0].mean():.3f}")