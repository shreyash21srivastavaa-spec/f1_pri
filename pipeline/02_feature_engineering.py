
import pandas as pd
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent
results = pd.read_csv(DIR / "race_results.csv")
quali = pd.read_csv(DIR / "qualifying.csv")


def quali_time_to_seconds(t):
    if pd.isna(t) or t == "":
        return np.nan
    parts = str(t).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


for col in ["Q1", "Q2", "Q3"]:
    quali[col + "_sec"] = quali[col].apply(quali_time_to_seconds)


quali["best_quali_sec"] = quali[["Q1_sec", "Q2_sec", "Q3_sec"]].min(axis=1)


quali["pole_time"] = quali.groupby(["season", "round"])["best_quali_sec"].transform("min")
quali["quali_gap_to_pole"] = quali["best_quali_sec"] - quali["pole_time"]

df = results.merge(
    quali[["season", "round", "driverId", "qualifying_position", "quali_gap_to_pole"]],
    on=["season", "round", "driverId"],
    how="left",
)


df = df.sort_values(["season", "round"]).reset_index(drop=True)



df["points_finish"] = (df["positionOrder"] <= 10).astype(int)
df["dnf"] = (~df["status"].str.contains("Finished|\\+", regex=True, na=False)).astype(int)


def add_rolling_feature(df, group_col, source_col, window, new_col):
    df[new_col] = (
        df.groupby(group_col)[source_col]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df

df = add_rolling_feature(df, "driverId", "positionOrder", 5, "driver_avg_finish_last5")
df = add_rolling_feature(df, "driverId", "points", 5, "driver_avg_points_last5")
df = add_rolling_feature(df, "constructorId", "positionOrder", 5, "team_avg_finish_last5")
df = add_rolling_feature(df, "constructorId", "points", 5, "team_avg_points_last5")


df["driver_circuit_avg_finish"] = (
    df.groupby(["driverId", "circuitId"])["positionOrder"]
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=[0, 1], drop=True)
)


fill_cols = [
    "driver_avg_finish_last5", "driver_avg_points_last5",
    "team_avg_finish_last5", "team_avg_points_last5",
    "driver_circuit_avg_finish",
]
df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())
df["quali_gap_to_pole"] = df["quali_gap_to_pole"].fillna(df["quali_gap_to_pole"].median())
df["qualifying_position"] = df["qualifying_position"].fillna(df["grid"])

feature_cols = [
    "grid", "qualifying_position", "quali_gap_to_pole",
    "driver_avg_finish_last5", "driver_avg_points_last5",
    "team_avg_finish_last5", "team_avg_points_last5",
    "driver_circuit_avg_finish",
]

model_df = df[["season", "round", "driverId", "constructorId", "circuitId"] + feature_cols + ["positionOrder", "points_finish", "dnf"]]
model_df.to_csv(DIR / "model_data.csv", index=False)
print(f"Saved model_data.csv with {len(model_df)} rows and features: {feature_cols}")
