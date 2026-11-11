
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

DIR = Path(__file__).resolve().parent
df = pd.read_csv(DIR / "model_data.csv")

feature_cols = [
    "grid", "qualifying_position", "quali_gap_to_pole",
    "driver_avg_finish_last5", "driver_avg_points_last5",
    "team_avg_finish_last5", "team_avg_points_last5",
    "driver_circuit_avg_finish",
]
target_col = "points_finish"


last_season = df["season"].max()
train = df[df["season"] < last_season]
test = df[df["season"] == last_season]

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

print(f"Train: {len(train)} rows (seasons {train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test)} rows (season {last_season})")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
logreg_preds = logreg.predict(X_test_scaled)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression (baseline) ---")
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print("ROC-AUC:", roc_auc_score(y_test, logreg_proba))
print(classification_report(y_test, logreg_preds))


rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print(classification_report(y_test, rf_preds))


importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:\n", importances)


joblib.dump(rf, DIR / "f1_model.joblib")
joblib.dump(scaler, DIR / "scaler.joblib")
importances.to_csv(DIR / "feature_importance.csv")
print("\nSaved f1_model.joblib, scaler.joblib, feature_importance.csv")
