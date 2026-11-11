
import requests
import pandas as pd
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
BASE = "https://api.jolpi.ca/ergast/f1"
SEASONS = list(range(2015, 2025))


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            print(f"Request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise Exception(f"Failed after {retries} retries: {url}")

def fetch_race_results(season):
    rows = []
    offset = 0
    limit = 100
    while True:
        url = f"{BASE}/{season}/results.json?limit={limit}&offset={offset}"
        data = get_json(url)
        mrdata = data["MRData"]
        total = int(mrdata["total"])
        races = mrdata["RaceTable"]["Races"]

        for race in races:
            round_num = int(race["round"])
            for res in race["Results"]:
                rows.append({
                    "season": season,
                    "round": round_num,
                    "raceName": race["raceName"],
                    "circuitId": race["Circuit"]["circuitId"],
                    "driverId": res["Driver"]["driverId"],
                    "constructorId": res["Constructor"]["constructorId"],
                    "grid": int(res["grid"]),
                    "position": res["position"],
                    "positionOrder": int(res["position"]),
                    "points": float(res["points"]),
                    "status": res["status"],
                    "laps": int(res["laps"]),
                })

        offset += limit
        if offset >= total:
            break
        time.sleep(1.0)
    return rows

def fetch_qualifying(season):
    rows = []
    offset = 0
    limit = 100
    while True:
        url = f"{BASE}/{season}/qualifying.json?limit={limit}&offset={offset}"
        data = get_json(url)
        mrdata = data["MRData"]
        total = int(mrdata["total"])
        races = mrdata["RaceTable"]["Races"]

        for race in races:
            round_num = int(race["round"])
            for res in race.get("QualifyingResults", []):
                rows.append({
                    "season": season,
                    "round": round_num,
                    "driverId": res["Driver"]["driverId"],
                    "constructorId": res["Constructor"]["constructorId"],
                    "qualifying_position": int(res["position"]),
                    "Q1": res.get("Q1"),
                    "Q2": res.get("Q2"),
                    "Q3": res.get("Q3"),
                })

        offset += limit
        if offset >= total:
            break
        time.sleep(1.0)
    return rows

if __name__ == "__main__":
    all_results = []
    all_quali = []

    for season in SEASONS:
        print(f"Fetching {season} race results...")
        all_results.extend(fetch_race_results(season))
        print(f"Fetching {season} qualifying...")
        all_quali.extend(fetch_qualifying(season))

    pd.DataFrame(all_results).to_csv(DIR / "race_results.csv", index=False)
    pd.DataFrame(all_quali).to_csv(DIR / "qualifying.csv", index=False)
    print("Done. Saved race_results.csv and qualifying.csv")
