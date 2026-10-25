# CineMatch — Personalized Movie Recommendation System

Academic project based on the supplied HCLTech Career Shaper Programme brief (Project 28).

## Features
- Flask web application for selecting a user and recommendation strategy.
- User-based collaborative filtering, item-based collaborative filtering, SVD matrix factorisation, and a hybrid display mode.
- Popular movies and recent rating history.
- JSON endpoint: `/api/recommendations/<user_id>?method=hybrid`.
- Data cleaning, documented data dictionary, three EDA charts, and hold-out evaluation (RMSE and Precision@5).

## Quick start (Windows)
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended):
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Generate EDA charts and evaluation metrics:
   ```powershell
   python eda.py
   python evaluate.py
   ```
6. Start the web application:
   ```powershell
   python app.py
   ```
7. Open `http://127.0.0.1:5000`.

## Files
- `data/dataset.csv`: cleaned data derived from the supplied 5,000-row file.
- `data/data_dictionary.csv`: descriptions of all dataset columns.
- `eda.py`: creates three visualisations under `outputs/`.
- `evaluate.py`: 80/20 random hold-out evaluation; exports `outputs/evaluation_metrics.csv`.
- `app.py`, `templates/index.html`: recommendation application.
- `Project_Report.docx`: structured report draft; review and personalize before submission.
- `Demo_Script.txt`: 1–2 minute demo outline.
- `Ethical_Compliance_Declaration.txt`: declaration template; learner must review and sign it personally.

## Important dataset limitation
The provided file has 5,000 rows but only **8 distinct movie titles**. It includes repeated user/movie pairs; preprocessing keeps the latest timestamp for each user/movie pair, leaving 2,868 ratings. This can reduce recommendation coverage and makes evaluation results unstable. State this limitation in your report and, if your mentor permits, expand the catalogue with a larger, properly sourced dataset such as MovieLens. Do not present synthetic ratings as real user behaviour.

## Method notes
The SVD implementation uses scikit-learn `TruncatedSVD` on a user-item matrix filled with the training mean. It is a lightweight educational matrix-factorisation approximation, not the Surprise library's biased SVD implementation. The evaluation script reports RMSE and Precision@5; Precision@5 defines a relevant held-out item as rating >= 4.0. A random split can have users/items with little overlap, so metrics should be interpreted cautiously.

## Before submission
- Run the scripts locally and inspect `outputs/evaluation_metrics.csv`.
- Add your name, roll number, mentor details and institution-specific front matter to the report.
- Check the report against your mentor's requirements and include screenshots from your own running app.
- Record a 1–2 minute demo.
- Review and sign the ethical declaration yourself.
