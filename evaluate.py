"""Hold-out evaluation for user/item collaborative filtering and SVD.

Metrics are computed on a random 80/20 split. Precision@K treats held-out
ratings >= 4.0 as relevant. With only eight unique movies, metrics are
illustrative and should not be interpreted as production-grade evidence.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "dataset.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)
global_mean = train.rating.mean()
mat = train.pivot_table(index="user_id", columns="movie_title", values="rating", aggfunc="mean")
filled = mat.fillna(global_mean)
movies = list(mat.columns)
users = list(mat.index)
user_sim = pd.DataFrame(cosine_similarity(filled), index=users, columns=users)
item_sim = pd.DataFrame(cosine_similarity(filled.T), index=movies, columns=movies)
k_svd = max(1, min(5, min(filled.shape)-1))
svd = TruncatedSVD(n_components=k_svd, random_state=42)
latent = svd.fit_transform(filled)
recon = pd.DataFrame(latent @ svd.components_, index=users, columns=movies)

def predict(row, method):
    uid, movie = int(row.user_id), row.movie_title
    if method == "svd":
        if uid in recon.index and movie in recon.columns: return float(recon.loc[uid, movie])
        return float(global_mean)
    if method == "user":
        if uid not in mat.index or movie not in mat.columns: return float(global_mean)
        ratings = mat[movie].dropna()
        if uid in ratings.index: ratings = ratings.drop(uid)
        sims = user_sim.loc[uid, ratings.index] if uid in user_sim.index else pd.Series(dtype=float)
        if not len(sims) or np.abs(sims).sum() == 0: return float(global_mean)
        return float((ratings * sims).sum() / np.abs(sims).sum())
    if method == "item":
        if uid not in mat.index or movie not in mat.columns: return float(global_mean)
        ur = mat.loc[uid].dropna()
        if movie in ur.index: ur = ur.drop(movie)
        sims = item_sim.loc[movie, ur.index] if movie in item_sim.index else pd.Series(dtype=float)
        if not len(sims) or np.abs(sims).sum() == 0: return float(global_mean)
        return float((ur * sims).sum() / np.abs(sims).sum())
    return float(global_mean)

def precision_at_k(method, k=5):
    # For each test user, rank catalogue items not present in train.
    vals = []
    for uid, group in test.groupby("user_id"):
        if uid not in mat.index: continue
        already = set(train.loc[train.user_id == uid, "movie_title"])
        candidates = [m for m in movies if m not in already]
        if not candidates: continue
        ranked = sorted(candidates, key=lambda m: predict(pd.Series({"user_id":uid,"movie_title":m}), method), reverse=True)[:k]
        relevant = set(group.loc[group.rating >= 4.0, "movie_title"])
        vals.append(sum(m in relevant for m in ranked) / max(1, len(ranked)))
    return float(np.mean(vals)) if vals else 0.0

rows=[]
for method in ("user","item","svd"):
    pred=np.array([np.clip(predict(r,method),1,5) for r in test.itertuples(index=False)])
    rmse=float(np.sqrt(mean_squared_error(test.rating, pred)))
    rows.append({"method":method,"RMSE":round(rmse,4),"Precision@5":round(precision_at_k(method,5),4)})
results=pd.DataFrame(rows)
results.to_csv(BASE / "outputs" / "evaluation_metrics.csv", index=False)
print(results.to_string(index=False))
print("\\nInterpretation: lower RMSE is better; higher Precision@5 is better. Treat results cautiously because the catalogue is very small and data may be synthetic.")
