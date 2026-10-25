from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "dataset.csv"
app = Flask(__name__)

def load_data():
    df = pd.read_csv(DATA)
    df["user_id"] = df["user_id"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df.dropna(subset=["user_id", "movie_title", "rating"])

RATINGS = load_data()
MATRIX = RATINGS.pivot_table(index="user_id", columns="movie_title", values="rating", aggfunc="mean")
MOVIE_INFO = RATINGS.groupby("movie_title").agg(
    genre=("genre", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    language=("language", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    release_year=("release_year", "median"),
    avg_rating=("rating", "mean"),
    rating_count=("rating", "count")
)
GLOBAL_MEAN = float(RATINGS["rating"].mean())

def popular_movies(n=8):
    info = MOVIE_INFO.copy()
    info["score"] = info["avg_rating"] * (info["rating_count"] / (info["rating_count"] + 5))
    return info.sort_values(["score", "rating_count"], ascending=False).head(n)

def recommend(user_id, method="hybrid", n=5):
    watched = set(RATINGS.loc[RATINGS.user_id == user_id, "movie_title"])
    candidates = [m for m in MATRIX.columns if m not in watched]
    if not candidates:
        candidates = list(MATRIX.columns)
    scores = {}
    user_known = user_id in MATRIX.index
    if method in ("user", "hybrid") and user_known:
        user_sim = pd.Series(
            cosine_similarity(MATRIX.fillna(0))[MATRIX.index.get_loc(user_id)],
            index=MATRIX.index
        ).drop(index=user_id, errors="ignore")
        for movie in candidates:
            vals = MATRIX[movie].dropna()
            common = vals.index.intersection(user_sim.index)
            if len(common):
                weights = user_sim.loc[common]
                denom = np.abs(weights).sum()
                scores[movie] = float((vals.loc[common] * weights).sum() / denom) if denom else GLOBAL_MEAN
    if method in ("item", "hybrid") and user_known:
        user_ratings = MATRIX.loc[user_id].dropna()
        if len(user_ratings):
            item_sim = pd.DataFrame(cosine_similarity(MATRIX.fillna(0).T), index=MATRIX.columns, columns=MATRIX.columns)
            for movie in candidates:
                if movie not in scores or method == "item":
                    sims = item_sim.loc[movie, user_ratings.index]
                    denom = np.abs(sims).sum()
                    scores[movie] = float((sims * user_ratings).sum() / denom) if denom else GLOBAL_MEAN
    # SVD-based latent-factor estimate for the matrix-factorisation option / hybrid fallback.
    if method in ("svd", "hybrid"):
        filled = MATRIX.fillna(GLOBAL_MEAN)
        try:
            k = max(1, min(5, min(filled.shape)-1))
            if k >= 1 and min(filled.shape) > 1:
                model = TruncatedSVD(n_components=k, random_state=42)
                latent = model.fit_transform(filled)
                reconstructed = latent @ model.components_
                row = MATRIX.index.get_loc(user_id) if user_known else None
                for movie in candidates:
                    col = list(MATRIX.columns).index(movie)
                    pred = float(reconstructed[row, col]) if row is not None else GLOBAL_MEAN
                    if method == "svd" or movie not in scores:
                        scores[movie] = pred
        except Exception:
            pass
    if not scores:
        for movie in candidates:
            scores[movie] = float(MOVIE_INFO.loc[movie, "avg_rating"])
    rows = []
    for movie, score in scores.items():
        info = MOVIE_INFO.loc[movie]
        rows.append({"movie_title": movie, "predicted_rating": round(float(np.clip(score, 1, 5)), 2),
                     "genre": info.genre, "language": info.language,
                     "release_year": int(info.release_year), "avg_rating": round(float(info.avg_rating), 2),
                     "rating_count": int(info.rating_count)})
    return sorted(rows, key=lambda x: x["predicted_rating"], reverse=True)[:n]

@app.route("/", methods=["GET", "POST"])
def index():
    users = sorted(RATINGS.user_id.unique().tolist())
    selected_user = request.form.get("user_id", str(users[0] if users else 1))
    method = request.form.get("method", "hybrid")
    try:
        uid = int(selected_user)
    except ValueError:
        uid = users[0]
    recs = recommend(uid, method=method, n=8)
    history = RATINGS[RATINGS.user_id == uid].sort_values("timestamp", ascending=False).head(10).to_dict("records")
    stats = {"ratings": len(RATINGS), "users": RATINGS.user_id.nunique(), "movies": RATINGS.movie_title.nunique(),
             "mean": round(RATINGS.rating.mean(), 2)}
    popular = popular_movies(8).reset_index().to_dict("records")
    return render_template("index.html", users=users, selected_user=uid, method=method,
                           recs=recs, history=history, stats=stats, popular=popular)

@app.route("/api/recommendations/<int:user_id>")
def api_recommendations(user_id):
    method = request.args.get("method", "hybrid")
    if method not in ("user", "item", "svd", "hybrid"):
        return {"error": "method must be user, item, svd, or hybrid"}, 400
    return {"user_id": user_id, "method": method, "recommendations": recommend(user_id, method, 5)}

if __name__ == "__main__":
    app.run(debug=True)
