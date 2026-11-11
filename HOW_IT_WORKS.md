# F1 top-10 predictor: how it is made, and what to learn from it

## The files
- `f1_top10_predictor.html` - the page. One file, no server. Open it in a browser.
- `build_data.py` - turns your model data into the numbers the page shows.
- Your original project (`01_fetch_data.py`, `02_feature_engineering.py`, `03_train_model.py`, CSVs) is where the data and model come from.

## The pipeline, start to finish
1. **Fetch** (`01_fetch_data.py`): pulls 2015-2024 race results and qualifying from the Jolpica API into CSVs.
2. **Features** (`02_feature_engineering.py`): one row per driver per race. Inputs: grid, qualifying position, gap to pole, driver and team recent form, driver's history at the circuit. Target: did they finish P10 or better (`points_finish`).
3. **Train** (`03_train_model.py`): Logistic Regression baseline and a Random Forest (300 trees, depth 6).
4. **Backtest** (`build_data.py`): for each year 2018-2024, train on earlier years only, predict that year, compare with the grid-only guess. Also exports 2024 probabilities per driver.
5. **Page**: the output JSON is pasted into the HTML as `const D = {...}`. JavaScript draws everything from that object.

## How the page works
- **Data**: `D.races` (24 races, each with drivers: team, grid, probability, finish), `D.wf` (accuracy per season: model `m`, grid-only `b`), `D.imp` (feature importance).
- **Draw**: `draw()` rebuilds the starting grid for the selected race. `shown` is true after you press Reveal. Correct calls get class `hit`, wrong ones `miss`. A call is correct when `(p >= 0.5) == (finish <= 10)`.
- **Banner ring**: a CSS `conic-gradient` animated with a registered CSS property (`@property --p`).
- **Charts**: the season bar chart is hand-built SVG, no library.
- **Light and dark mode**: colours are CSS variables on `:root`, swapped inside `prefers-color-scheme: dark`.
- **To update the numbers**: rerun `build_data.py`, then replace the `const D=...;` line in the HTML with the new `data.json` content.

## What to learn from it
1. **Always compare with a dumb baseline.** "Starts top 10, finishes top 10" gets 83.7% in 2024. Your model got 84.1-84.8%. Without that comparison, 84% sounds great and means little.
2. **Split by time, not randomly.** Races are ordered in time, so train on the past and test on the future. A random split lets the model peek.
3. **Test on many years, not one.** One season (479 rows) is noisy. Walk-forward over 2018-2024 gave a steadier picture: about +2.5 points on average.
4. **Leakage hides in rolling features.** `groupby(team).shift(1)` let the second teammate see the first teammate's result from the same race. Fix: aggregate per race first, then lag.
5. **Fill missing values using training data only.** Medians computed on the whole dataset leak a little test information.
6. **Pick metrics that match the question.** Accuracy answers "how often right at 50%". ROC-AUC answers "does it rank scorers above non-scorers". Report both and say what they mean.
7. **Pin library versions.** The saved `.joblib` was made with scikit-learn 1.6.1 and warns on newer versions. Add a `requirements.txt`.
8. **Presenting results is part of the work.** A small honest edge, clearly explained, is more convincing than an inflated number.

## How to explain it in an interview (30 seconds)
"I built a model that predicts whether an F1 driver finishes in the points, using qualifying and recent form. I tested it year by year on seasons it had not seen and compared it with a simple rule. It beats the rule by about 2.5 points on average. I also found and fixed a leak in my team-form features. The page lets you pick a 2024 race, see the odds, and reveal what happened."

## Limits and next steps
- Not modelled: weather, tyres, penalties after qualifying, breakdowns.
- Next steps worth doing: a calibration check (do "70%" calls come true about 70% of the time?), predicting finishing position instead of just top 10, and a `requirements.txt` plus README in your repo.
