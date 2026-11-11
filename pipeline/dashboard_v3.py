import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="F1 Race Predictor", layout="wide", page_icon="🏎️")

F1_RED = "#E10600"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{ font-family: 'Titillium Web', sans-serif; }}
    .main {{ background-color: #0a0a0c; }}
    h1 {{ font-size: 38px !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: 1px; }}
    h2, h3 {{ font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.5px; }}
    p, li, .stMarkdown {{ font-size: 15px !important; line-height: 1.6 !important; }}

    .red-underline {{ height: 4px; background: {F1_RED}; width: 60px; margin: 4px 0 20px 0; }}

    .fact-card {{
        background-color: #17171b; border-radius: 8px; padding: 18px 22px;
        margin-bottom: 12px; border-left: 4px solid {F1_RED};
    }}
    .fact-card .title {{ font-size: 17px; font-weight: 700; color: #fafafa; }}
    .fact-card .pct {{ font-size: 19px; font-weight: 900; color: {F1_RED}; float: right; }}
    .fact-card .desc {{ font-size: 13px; color: #9199a3; margin-top: 8px; }}

    .hero {{
        background: #111114; border-radius: 6px; padding: 30px 34px; margin-bottom: 22px;
        border-top: 4px solid {F1_RED};
    }}
    .simple-box {{
        background-color: #17171b; border-radius: 8px; padding: 18px 20px; margin-bottom: 10px;
        border-top: 3px solid #333338;
    }}

    .podium-card {{
        background-color: #17171b; border-radius: 8px; padding: 18px; text-align: center;
    }}
    .podium-rank {{ font-size: 13px; font-weight: 900; color: #6b6f76; letter-spacing: 2px; }}
    .podium-name {{ font-size: 20px; font-weight: 800; color: #fafafa; margin: 6px 0 2px 0; }}
    .podium-team {{ font-size: 12px; color: #9199a3; text-transform: uppercase; letter-spacing: 1px; }}
    .podium-pct {{ font-size: 22px; font-weight: 900; margin-top: 10px; }}

    section[data-testid="stSidebar"] {{ background-color: #111114; border-right: 2px solid {F1_RED}; }}
</style>
""", unsafe_allow_html=True)


TEAM_COLORS = {
    "mercedes": "#27F4D2", "red_bull": "#3671C6", "ferrari": "#E8002D",
    "mclaren": "#FF8000", "alpine": "#2293D1", "aston_martin": "#229971",
    "williams": "#64C4FF", "haas": "#B6BABD", "sauber": "#52E252",
    "alfa": "#C92D4B", "force_india": "#F596C8", "racing_point": "#F596C8",
    "renault": "#FFF500", "toro_rosso": "#5E8FAA", "alphatauri": "#5E8FAA",
    "rb": "#6692FF", "lotus_f1": "#FFB800", "manor": "#6E6E6E",
    "marussia": "#6E0000", "caterham": "#016633", "toyota": "#CC0000",
    "brawn": "#B7D014", "honda": "#FFFFFF", "bmw_sauber": "#006EFF",
}
DEFAULT_TEAM_COLOR = "#7d7d7d"


def team_color(constructor_id):
    return TEAM_COLORS.get(constructor_id, DEFAULT_TEAM_COLOR)


def pretty(s):
    return str(s).replace("_", " ").title()



model = joblib.load(DIR / "f1_model.joblib")
importance = pd.read_csv(DIR / "feature_importance.csv")
importance.columns = ["feature", "importance_score"]
model_data = pd.read_csv(DIR / "model_data.csv")

FEATURE_COLS = [
    "grid", "qualifying_position", "quali_gap_to_pole",
    "driver_avg_finish_last5", "driver_avg_points_last5",
    "team_avg_finish_last5", "team_avg_points_last5",
    "driver_circuit_avg_finish",
]

LABEL_MAP = {
    "qualifying_position": "Qualifying Position",
    "grid": "Starting Grid Position",
    "driver_avg_points_last5": "Driver's Recent Form (Points)",
    "team_avg_points_last5": "Team's Recent Form (Points)",
    "quali_gap_to_pole": "Gap to Pole Time",
    "driver_avg_finish_last5": "Driver's Recent Form (Finish)",
    "team_avg_finish_last5": "Team's Recent Form (Finish)",
    "driver_circuit_avg_finish": "Driver's History at This Track",
}
EXPLAIN_MAP = {
    "qualifying_position": "Where the driver qualified — P1, P5, etc. In F1, track position is everything, so this is the single biggest clue.",
    "grid": "Actual starting slot on race day (can differ from qualifying due to penalties).",
    "driver_avg_points_last5": "Has this driver been scoring well in their last 5 races? Good recent form suggests they'll keep it up.",
    "team_avg_points_last5": "Is the car/team currently competitive? A driver's result depends heavily on the car underneath them.",
    "quali_gap_to_pole": "How many seconds slower than the fastest qualifying lap. A small gap means genuine race pace, not just a lucky grid slot.",
    "driver_avg_finish_last5": "Average finishing position over the last 5 races — a second view of current form.",
    "team_avg_finish_last5": "Same idea, but for the team as a whole across both cars.",
    "driver_circuit_avg_finish": "How this driver has historically done at this exact track — turns out this matters least; current form beats track history.",
}
importance["label"] = importance["feature"].map(LABEL_MAP)
importance["explanation"] = importance["feature"].map(EXPLAIN_MAP)
importance["pct"] = importance["importance_score"] / importance["importance_score"].sum()


st.sidebar.markdown(f"<h2 style='color:white;'>F1 ANALYTICS</h2><div class='red-underline'></div>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "What Matters Most", "Driver Form", "Race Predictor"], label_visibility="collapsed")




if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Can we predict who scores points before the race even starts?</h1>
        <p>A machine learning model trained on 10 years of real Formula 1 races (2015–2024),
        predicting whether a driver finishes in the <b>top 10</b> — using only information
        known <b>before</b> the race begins.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="simple-box"><h3>What it predicts</h3><p>The chance each driver finishes in the top 10, for any real race from the last 10 seasons.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="simple-box"><h3>What it learned from</h3><p>Starting grid, qualifying pace, and how well the driver/team have been doing recently.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="simple-box"><h3>How good is it</h3><p>Right about 86% of the time, tested on a full season it had never seen before.</p></div>', unsafe_allow_html=True)

    st.markdown("### Why starting position matters so much")
    st.markdown("<div class='red-underline'></div>", unsafe_allow_html=True)
    st.write("Every dot is one driver, in one real race. Left to right is where they started. Up and down is where they finished. Green dots cluster near the front — starting near the front usually means finishing near the front too.")

    sample = model_data.sample(min(1500, len(model_data)), random_state=1)
    fig = px.scatter(
        sample, x="grid", y="positionOrder", color="points_finish",
        color_discrete_map={1: "#27F4D2", 0: "#3a3a40"},
        labels={"grid": "Starting Grid Position", "positionOrder": "Actual Finish Position", "points_finish": "Scored Points"},
        opacity=0.65,
        render_mode="svg"
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c", legend_title_text="Scored Points")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)




elif page == "What Matters Most":
    st.title("What Drives a Points Finish")
    st.markdown("<div class='red-underline'></div>", unsafe_allow_html=True)
    st.write("Eight factors, weighted by how much the model actually relies on each one.")

    importance_sorted = importance.sort_values(by="importance_score", ascending=True)
    fig = px.bar(
        importance_sorted, x="importance_score", y="label", orientation="h",
        color="importance_score", color_continuous_scale=["#3a3a40", F1_RED],
        labels={"importance_score": "Importance", "label": ""},
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c", coloraxis_showscale=False, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What each factor means")
    for _, row in importance.sort_values(by="importance_score", ascending=False).iterrows():
        st.markdown(f"""
        <div class="fact-card">
            <span class="title">{row['label']}</span>
            <span class="pct">{row['pct']:.0%}</span>
            <div style="clear:both;"></div>
            <div class="desc">{row['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)




elif page == "Driver Form":
    st.title("Driver Form Over Time")
    st.markdown("<div class='red-underline'></div>", unsafe_allow_html=True)
    st.write("Compare up to 3 drivers. Lines are colored by each driver's most-raced team.")

    driver_list = sorted(model_data["driverId"].unique())
    selected_drivers = st.multiselect("Drivers", driver_list, default=[driver_list[0]], max_selections=3)

    if not selected_drivers:
        st.warning("Pick at least one driver.")
    else:
        subset = model_data[model_data["driverId"].isin(selected_drivers)].copy()
        subset = subset.sort_values(by=["season", "round"])
        subset["race_label"] = subset["season"].astype(str) + " R" + subset["round"].astype(str)



        full_order = (
            model_data[["season", "round"]].drop_duplicates().sort_values(by=["season", "round"])
        )
        full_order["race_label"] = full_order["season"].astype(str) + " R" + full_order["round"].astype(str)
        category_order = full_order["race_label"].tolist()

        metric_choice = st.radio("Show", ["Average Points (last 5 races)", "Average Finish Position (last 5 races)"], horizontal=True)
        y_col = "driver_avg_points_last5" if "Points" in metric_choice else "driver_avg_finish_last5"

        fig = go.Figure()
        for d in selected_drivers:
            d_data = subset[subset["driverId"] == d]
            dominant_team = d_data["constructorId"].mode().iloc[0]
            fig.add_trace(go.Scatter(
                x=d_data["race_label"], y=d_data[y_col], mode="lines",
                name=pretty(d), line=dict(color=team_color(dominant_team), width=2.5),
            ))

        fig.update_xaxes(categoryorder="array", categoryarray=category_order, tickangle=-60, tickfont=dict(size=8))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c",
            height=460, yaxis_title=metric_choice, xaxis_title="Race",
        )
        if "Finish" in metric_choice:
            fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower finish position is better (P1 = best). Higher points is better.")




elif page == "Race Predictor":
    st.title("Race Outcome Predictor")
    st.markdown("<div class='red-underline'></div>", unsafe_allow_html=True)
    st.write("Pick a real race. The model guesses each driver's chance of scoring points, using only what was known before the race — then compare it to what actually happened.")

    col_a, col_b = st.columns(2)
    with col_a:
        season_list = sorted(model_data["season"].unique(), reverse=True)
        selected_season = st.selectbox("Season", season_list)
    with col_b:
        rounds_in_season = sorted(model_data[model_data["season"] == selected_season]["round"].unique())
        selected_round = st.selectbox("Round", rounds_in_season)

    race_data = model_data[
        (model_data["season"] == selected_season) & (model_data["round"] == selected_round)
    ].copy()

    if race_data.empty:
        st.warning("No data for this race.")
    else:
        X_race = race_data[FEATURE_COLS]
        race_data["raw_prob"] = model.predict_proba(X_race)[:, 1]
        race_data["prob_pct"] = race_data["raw_prob"] * 100
        race_data["Driver"] = race_data["driverId"].apply(pretty)
        race_data["Team"] = race_data["constructorId"].apply(pretty)
        race_data["team_color"] = race_data["constructorId"].apply(team_color)
        race_data = race_data.sort_values(by="prob_pct", ascending=False)

        st.markdown("### Model's Top 3 Picks")
        top3 = race_data.head(3).reset_index(drop=True)
        cols = st.columns(3)
        for i, col in enumerate(cols):
            row = top3.loc[i]
            with col:
                st.markdown(f"""
                <div class="podium-card" style="border-top: 4px solid {row['team_color']};">
                    <div class="podium-rank">P{i+1} PREDICTED</div>
                    <div class="podium-name">{row['Driver']}</div>
                    <div class="podium-team">{row['Team']}</div>
                    <div class="podium-pct" style="color:{row['team_color']};">{row['prob_pct']:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Full Grid Predictions")
        fig = go.Figure(go.Bar(
            x=race_data["prob_pct"], y=race_data["Driver"], orientation="h",
            marker_color=race_data["team_color"],
            text=race_data["prob_pct"].round(0).astype(int).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c",
            height=600, xaxis_title="Predicted Chance of Scoring Points (%)", yaxis_title="",
            yaxis=dict(categoryorder="total ascending"), xaxis_range=[0, 110],
        )
        st.plotly_chart(fig, use_container_width=True)

        display_df = race_data[["Driver", "Team", "grid", "prob_pct", "points_finish", "positionOrder"]].rename(columns={
            "grid": "Grid", "prob_pct": "Predicted Chance", "points_finish": "Scored Points", "positionOrder": "Actual Finish",
        })
        display_df["Scored Points"] = display_df["Scored Points"].map({1: "Yes", 0: "No"})

        st.dataframe(
            display_df,
            column_config={"Predicted Chance": st.column_config.ProgressColumn("Predicted Chance", format="%.0f%%", min_value=0, max_value=100)},
            use_container_width=True, hide_index=True,
        )

        correct = (
            ((race_data["raw_prob"] >= 0.5) & (race_data["points_finish"] == 1)) |
            ((race_data["raw_prob"] < 0.5) & (race_data["points_finish"] == 0))
        ).sum()
        total = len(race_data)
        st.success(f"The model correctly called {correct} of {total} drivers in this race ({correct/total:.0%} accurate).")