#PER MODEL FEWSHOT STATS TEST

#libraries
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# LOAD DATA
df0 = pd.read_excel(
    r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\results_spreadsheets\results_fewshot_vs_zeroshot.xlsx'
)


# CLEAN THE DATA
df = df0.copy()
for col in ["Source","Test","Model"]:
    df[col] = df[col].astype(str).str.strip()

df["Source"] = df["Source"].str.capitalize()
df["Test"]   = df["Test"].str.capitalize().replace({"Wiki":"Wikipedia"})
PLATFORMS = {"Gab","Reddit","Wikipedia"}

name_map = {"rf":"RF","randomforest":"RF","svm":"SVM","logreg":"LR","lr":"LR",
            "cnn":"CNN","lstm":"RNN","rnn":"RNN","bert":"BERT","knn":"kNN"}
df["Model"] = df["Model"].str.lower().map(name_map).fillna(df["Model"])





# PERMUTATON TEST
def signflip_perm_mean(deltas, n_perm=100_000, alternative="greater"):
    deltas = np.asarray(deltas, float)
    obs = deltas.mean()
    signs = rng.choice([1, -1], size=(n_perm, deltas.size))
    null = (np.where(signs==1, deltas, -deltas)).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


#BOOTSTRAPS
def bootstrap_mean_ci(deltas, n=10000, alpha=0.05):
    deltas = np.asarray(deltas, float)
    if len(deltas) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi



#build per source, model, target few-shot improvements

def per_target_improvements(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    columns_needed = {"Source","Model","Test",f"{metric}_few",f"{metric}_zero"}
    miss = columns_needed - set(frame.columns)
    if miss:
        raise ValueError(f"Missing columns for metric {metric}: {sorted(miss)}")

    sub = frame[["Source","Model","Test",f"{metric}_few",f"{metric}_zero"]].copy()

    sub = sub[sub["Test"].isin(PLATFORMS)]

    agg = (sub.groupby(["Source","Model","Test"], as_index=False)
              .agg({f"{metric}_few":"mean", f"{metric}_zero":"mean"}))
    
    agg["delta"] = agg[f"{metric}_few"] - agg[f"{metric}_zero"]

    return agg  # columns: Source, Model, Test, {few,zero}, delta



#EVALATE LOOP

metrics = ["F1","MCC","AUC"]
rows = []

for metric in metrics:
    unit = per_target_improvements(df, metric)  #per Source, Model, Target
    for model, sub in unit.groupby("Model"):
        deltas = sub["delta"].to_numpy()
        obs, p = signflip_perm_mean(deltas, n_perm=100_000, alternative="greater")
        lo, hi = bootstrap_mean_ci(deltas, n=10000, alpha=0.05)
        rows.append({
            "Metric": metric,
            "Model": model,
            "mean_Δ(few−zero)": obs,
            "CI95_low": lo,
            "CI95_high": hi,
            "p_perm": p,
            "n_pairs": len(deltas)
        })



#PRINT THE RESULTS
per_model_fewshot = pd.DataFrame(rows).sort_values(["Metric","Model"]).reset_index(drop=True)
print(per_model_fewshot.to_string(index=False))
