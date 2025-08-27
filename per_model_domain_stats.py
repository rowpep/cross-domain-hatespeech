#PER MODEL IN DOMAIN STATS


#libraries
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

#LOAD DATA
domain_df = pd.read_excel(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\results_spreadsheets\results_indomain_vs_crossdomain.xlsx')

#CLEAN THE DATA
df = domain_df.copy()
df["Source"] = df["Source"].astype(str).str.strip().str.capitalize()
df["Test"]   = df["Test"].astype(str).str.strip().str.capitalize()
PLATFORMS = {"Gab","Reddit","Wikipedia"}



#per source target and model differences 

def per_target_deltas(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    # in domain score per source/model
    indom = (frame[(frame["DomainType"]=="InDomain") & (frame["Test"]==frame["Source"])]
             .groupby(["Source","Model"], as_index=False)[metric].mean()
             .rename(columns={metric:"IN"}))
    
    # cross-domain score per source, model, target
    xdom = (frame[(frame["DomainType"]=="CrossDomain") & (frame["Test"].isin(PLATFORMS))]
            .groupby(["Source","Model","Test"], as_index=False)[metric].mean()
            .rename(columns={metric:"CD"}))
    
    # merge and calculate delta for each source, model, target
    paired = xdom.merge(indom, on=["Source","Model"], how="inner")
    paired["delta"] = paired["IN"] - paired["CD"]
    return paired  # columns: Source, Model, Test (target), IN, CD, delta



#PERMUTATION TESTING
def signflip_perm_mean(deltas, n_perm=100_000, alternative="greater"):
    deltas = np.asarray(deltas, float)
    obs = deltas.mean()
    signs = rng.integers(0, 2, size=(n_perm, deltas.size)) * 2 - 1
    null = (signs * deltas).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p



#BOOTSTRAP
def bootstrap_mean_ci(deltas, n=10000, alpha=0.05):
    deltas = np.asarray(deltas, float)
    if len(deltas) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

#per model evaluation loop
rows = []
for metric in ["F1","MCC","AUC"]:
    units = per_target_deltas(df, metric) 
    for model, sub in units.groupby("Model"):
        deltas = sub["delta"].values
        IN_mean = sub["IN"].mean()  
        CD_mean = sub["CD"].mean()
        d_mean, p = signflip_perm_mean(deltas, n_perm=100_000, alternative="greater")
        lo, hi = bootstrap_mean_ci(deltas)
        rows.append({
            "Metric": metric,
            "Model": model,
            "IN_mean": IN_mean,
            "CD_mean": CD_mean,
            "Delta_mean": d_mean,
            "Delta_CI95_low": lo,
            "Delta_CI95_high": hi,
            "p_perm": p,
            "n_sources": len(deltas)
        })


#PRINT RESULTS
per_model_results = pd.DataFrame(rows).sort_values(["Metric","Model"])
print(per_model_results)
