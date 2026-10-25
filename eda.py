"""Exploratory data analysis: writes three PNG charts to outputs/."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "dataset.csv")
out = BASE / "outputs"
out.mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating"); plt.ylabel("Number of ratings"); plt.tight_layout()
plt.savefig(out / "rating_distribution.png", dpi=160); plt.close()

plt.figure(figsize=(9, 5))
df["genre"].value_counts().sort_values().plot(kind="barh")
plt.title("Ratings by Genre")
plt.xlabel("Number of ratings"); plt.ylabel("Genre"); plt.tight_layout()
plt.savefig(out / "genre_counts.png", dpi=160); plt.close()

plt.figure(figsize=(9, 5))
top = df["movie_title"].value_counts().head(10).sort_values()
top.plot(kind="barh")
plt.title("Most Frequently Rated Movies")
plt.xlabel("Number of ratings"); plt.ylabel("Movie"); plt.tight_layout()
plt.savefig(out / "movie_popularity.png", dpi=160); plt.close()

print("Rows:", len(df), "| users:", df.user_id.nunique(), "| movies:", df.movie_title.nunique())
print("Mean rating:", round(df.rating.mean(), 3))
print("Charts saved in outputs/")
