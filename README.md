# F1 top-10 predictor

Predicts whether an F1 driver finishes in the points (top 10), tested year by year on 2018-2024.

## Just see the page
Double-click `index.html`. It works offline except for the Google Fonts, which fall back to system fonts.
To publish: drop `index.html` on GitHub Pages or Netlify.

## Run everything
```bash
pip install -r requirements.txt
cd pipeline
python build_data.py     # retrains, writes data.json
python update_page.py    # puts data.json into ../index.html
```
The CSVs are included, so you do not need to download anything.

## Optional: rebuild from scratch
```bash
cd pipeline
python 01_fetch_data.py            # downloads 2015-2024 data (slow, rate limited)
python 02_feature_engineering.py
python 03_train_model.py
streamlit run dashboard_v3.py      # your original Streamlit dashboard
```

## What is where
- `index.html` the web page (all data is inside it)
- `pipeline/` your original scripts and data, plus `build_data.py` and `update_page.py`
- `HOW_IT_WORKS.md` how it is built and what to learn from it

Note: `build_data.py` uses a fixed team-form feature (per-race, then lagged). `02_feature_engineering.py` still has the original teammate leak, so numbers from `03_train_model.py` differ slightly.
