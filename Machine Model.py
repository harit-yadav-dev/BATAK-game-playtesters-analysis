import os, re, json, math, glob
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,cross_val_score, RepeatedStratifiedKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = r"C:\Users\hk290\OneDrive\Desktop\Team 9785-S2-56-B\Playtesters"
OUTPUT_DIR = r"C:\Users\hk290\OneDrive\Desktop\Team 9785-S2-56-B\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NAME_AGE_RE = re.compile(r"^(?P<name>.+)_(?P<age>\d{1,3})\.json$", re.IGNORECASE)
RANDOM_STATE = 42
TOP_K = 4
ACC_TARGET = 0.80
EXCLUDE_TOP = {"duration_std"}
def _clean_list(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
def smean(xs): xs = _clean_list(xs); return float(mean(xs)) if xs else np.nan
def smedian(xs): xs = _clean_list(xs); return float(median(xs)) if xs else np.nan
def sstd(xs): xs = _clean_list(xs); return float(pstdev(xs)) if len(xs) > 1 else 0.0
def pretty_feature_name(raw):
    m = {
        "rt_mean": "Reaction Time (mean, s)",
        "rt_median": "Reaction Time (median, s)",
        "rt_std": "Reaction Time variability (std, s)",
        "press_count": "Number of presses",
        "correct_rate": "Accuracy (correct presses / total)",
        "left_ratio": "Left-hand usage ratio",
        "right_ratio": "Right-hand usage ratio",
        "hr_mean": "Heart rate (mean, bpm)",
        "hr_max": "Heart rate (max, bpm)",
        "hr_std": "Heart rate variability (std, bpm)",
        "duration_mean": "Session duration (mean, s)",
        "duration_std": "Session duration variability (std, s)",
        "exited_early_rate": "Exited early rate",
        "speed_mean": "Controller speed (mean, m/s)",
        "speed_median": "Controller speed (median, m/s)",
        "speed_std": "Controller speed variability (std, m/s)",
    }
    return m.get(raw, raw)

def path_len_and_duration(traj):
    if not traj or len(traj) < 2: return 0.0, 0.0
    t = sorted(traj, key=lambda z: float(z.get("timeStamp", 0.0)))
    d = 0.0
    for a, b in zip(t, t[1:]):
        pa, pb = a.get("position"), b.get("position")
        if not pa or not pb: continue
        dx = float(pb.get("x", 0.0)) - float(pa.get("x", 0.0))
        dy = float(pb.get("y", 0.0)) - float(pa.get("y", 0.0))
        dz = float(pb.get("z", 0.0)) - float(pa.get("z", 0.0))
        d += math.sqrt(dx*dx + dy*dy + dz*dz)
    t0 = float(t[0].get("timeStamp", 0.0))
    t1 = float(t[-1].get("timeStamp", 0.0))
    return d, max(0.0, t1 - t0)
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[SKIP] Bad JSON → {path}\n       {e}")
        return None

def extract_features(path):
    data = load_json(path)
    if data is None: return None
    sessions = data if isinstance(data, list) else [data]
    rt, corrects, hands, hr = [], [], [], []
    path_lengths, path_times = [], []
    durations, exited = [], []
    for sess in sessions:
        durations.append(float(sess.get("Duration", np.nan)))
        exited.append(bool(sess.get("ExitedEarly", False)))
        for p in (sess.get("PressData") or []):
            if "ReactionTime" in p: rt.append(float(p["ReactionTime"]))
            corrects.append(bool(p.get("CorrectPress", False)))
            if p.get("HandUsed") in ("left", "right"): hands.append(p["HandUsed"])
            pl, pt = path_len_and_duration(p.get("ControllerTrajectory", []) or [])
            path_lengths.append(pl); path_times.append(pt)
        for h in (sess.get("HeartrateData") or []):
            if "Heartrate" in h: hr.append(float(h["Heartrate"]))
    n = len(corrects)
    n_correct = sum(1 for c in corrects if c)
    n_left = sum(1 for h in hands if h == "left")
    n_right = sum(1 for h in hands if h == "right")
    speeds = [pl/pt for pl, pt in zip(path_lengths, path_times) if pt > 0]
    return {
        "rt_mean": round(smean(rt), 2),
        "rt_median": round(smedian(rt), 2),
        "rt_std": round(sstd(rt), 2),
        "press_count": int(n),
        "correct_rate": round((n_correct / n), 3) if n > 0 else np.nan,
        "left_ratio": round(n_left / max(1, n_left + n_right), 3),
        "right_ratio": round(n_right / max(1, n_left + n_right), 3),
        "hr_mean": round(smean(hr), 2),
        "hr_max": round(max(hr), 2) if hr else np.nan,
        "hr_std": round(sstd(hr), 2),
        "duration_mean": round(smean(durations), 2),
        "duration_std": round(sstd(durations), 2),
        "exited_early_rate": round((sum(exited) / len(exited)), 3) if exited else 0.0,
        "speed_mean": round(smean(speeds), 3),
        "speed_median": round(smedian(speeds), 3),
        "speed_std": round(sstd(speeds), 3),
    }

def list_name_age_files(folder):
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return []
    paths = glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    files = [p for p in paths if NAME_AGE_RE.match(os.path.basename(p))]
    print(f"[INFO] Using {len(files)} participant files from {folder}")
    if not files:
        try:
            cand = [os.path.basename(x) for x in paths[:20]]
            print("[DEBUG] Found JSONs (first 20):", cand)
            print("[DEBUG] Expecting filenames like 'Gurkeerat_18.json'")
        except Exception:
            pass
    return files
def build_dataset(folder):
    rows = []
    for path in list_name_age_files(folder):
        m = NAME_AGE_RE.match(os.path.basename(path))
        if not m: continue
        name, age = m.group("name"), int(m.group("age"))
        feats = extract_features(path)
        if feats is None: continue
        feats["Name"], feats["Age"] = name, age
        if 18 <= age <= 35:
            feats["AgeGroup"] = 0
        elif 36 <= age <= 50:
            feats["AgeGroup"] = 1
        else:
            print(f"[SKIP] Age out of target range → {os.path.basename(path)}")
            continue
        rows.append(feats)
    return pd.DataFrame(rows)
def save_feature_importance(top4_df, out_path):
    plt.figure(figsize=(6, 4))
    plt.barh(list(reversed(top4_df["Feature_pretty"].tolist())),
             list(reversed(top4_df["Importance"].tolist())))
    plt.title("Top 4 Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
def plot_group_feature(df, feature, label, out_path):
    g = df.groupby("AgeGroup")[feature].mean()
    labels = ["18–35", "36–50"]
    vals = [g.get(0, np.nan), g.get(1, np.nan)]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, vals)
    plt.title(label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
def save_group_accuracy(df, out_path):
    g = df.groupby("AgeGroup")["correct_rate"].mean()
    labels = ["18–35", "36–50"]
    vals = [g.get(0, np.nan), g.get(1, np.nan)]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, vals)
    plt.title("Average In-Game Accuracy by Age Group")
    plt.ylabel("Accuracy (correct_rate)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
def evaluate_with_cv(model, X, y, n_splits=5, n_repeats=5):
    mpc = int(y.value_counts().min())
    n_splits = min(n_splits, max(2, mpc))
    if n_splits < 2:
        print("[WARN] Too few samples per class for CV; skipping CV evaluation.")
        return np.array([])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=rskf, scoring="accuracy")
    print(f"\n=== Repeated CV ({n_splits}-fold x {n_repeats} repeats) ===")
    print(f"Mean accuracy: {scores.mean():.3f}  ± {scores.std():.3f}")
    print("Some fold scores:", np.round(scores[:min(10, len(scores))], 3))
    return scores

def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
def main():
    df = build_dataset(DATA_DIR)
    if df.empty: raise SystemExit("No valid data found.")
    feats_csv = os.path.join(OUTPUT_DIR, "playtest_features.csv")
    df.round(3).to_csv(feats_csv, index=False)
    print(f"[OK] Saved engineered features → {feats_csv}")

    X = df.drop(columns=["Name", "Age", "AgeGroup"])
    y = df["AgeGroup"].astype(int)
    X = X.loc[:, X.notna().any(axis=0)].fillna(X.mean(numeric_only=True))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(0.2, min(0.4, 1.0 / max(2, len(df)))),
        random_state=RANDOM_STATE, stratify=y
    )
    mct = y_train.value_counts().min()
    if mct >= 2:
        n_splits = min(3, int(mct))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = None

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    rf_grid = {"n_estimators": [150, 300], "max_depth": [None, 12, 24], "min_samples_split": [2, 5]}

    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                   class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000))
    ])
    logit_grid = {"clf__C": [0.25, 0.5, 1.0]}
    if cv is not None:
        rf_gs = GridSearchCV(rf, rf_grid, cv=cv, scoring="accuracy")
        rf_gs.fit(X_train, y_train)
        rf_cv_mean = rf_gs.best_score_
        rf_cv_std = rf_gs.cv_results_["std_test_score"][rf_gs.best_index_]
        logit_gs = GridSearchCV(logit, logit_grid, cv=cv, scoring="accuracy")
        logit_gs.fit(X_train, y_train)
        logit_cv_mean = logit_gs.best_score_
        logit_cv_std = logit_gs.cv_results_["std_test_score"][logit_gs.best_index_]
        if rf_cv_mean >= logit_cv_mean:
            best_model, best_name = rf_gs.best_estimator_, "RandomForest"
            best_cv_mean, best_cv_std, used_splits = rf_cv_mean, rf_cv_std, n_splits
        else:
            best_model, best_name = logit_gs.best_estimator_, "Logistic(L1)"
            best_cv_mean, best_cv_std, used_splits = logit_cv_mean, logit_cv_std, n_splits
    else:
        best_model, best_name = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            random_state=RANDOM_STATE, class_weight="balanced"
        ), "RandomForest(no-CV)"
        best_model.fit(X_train, y_train)
        best_cv_mean = best_cv_std = np.nan
        used_splits = 0

    cv_full_scores = evaluate_with_cv(best_model, X, y, n_splits=5, n_repeats=5)
    cv_full_mean = float(cv_full_scores.mean()) if cv_full_scores.size else np.nan
    cv_full_std = float(cv_full_scores.std()) if cv_full_scores.size else np.nan
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    if best_name.startswith("RandomForest"):
        importances = best_model.feature_importances_
    else:
        coef = np.abs(best_model.named_steps["clf"].coef_).ravel()
        importances = (coef / coef.sum()) if coef.sum() != 0 else np.zeros_like(coef)

    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    imp_df = imp_df[~imp_df["Feature"].isin(EXCLUDE_TOP)]
    top4 = imp_df.head(TOP_K).copy()
    top4["Feature_pretty"] = top4["Feature"].map(pretty_feature_name)

    top4.to_csv(os.path.join(OUTPUT_DIR, "top4_features.csv"), index=False)

    g = df.groupby("AgeGroup")["correct_rate"].mean()
    pd.DataFrame({
        "AgeGroup": ["18–35", "36–50"],
        "avg_correct_rate": [g.get(0, np.nan), g.get(1, np.nan)]
    }).to_csv(os.path.join(OUTPUT_DIR, "group_accuracy.csv"), index=False)

    players = df[["Name", "Age", "AgeGroup", "correct_rate"]].copy()
    players["AgeGroupLabel"] = players["AgeGroup"].map({0: "18–35", 1: "36–50"})
    players["correct_rate_pct"] = (players["correct_rate"] * 100).round(2)
    players = players.drop(columns=["correct_rate"]).sort_values(
        ["AgeGroup", "correct_rate_pct"], ascending=[True, False]
    )
    players.to_csv(os.path.join(OUTPUT_DIR, "player_accuracy.csv"), index=False)

    for grp, label in [(0, "18–35"), (1, "36–50")]:
        sub = players[players["AgeGroup"] == grp].head(3)
        if not sub.empty:
            print(f"\nTop performers — {label}:")
            for _, r in sub.iterrows():
                print(f"  {r['Name']} (Age {int(r['Age'])}) — {r['correct_rate_pct']:.2f}%")

    save_feature_importance(top4, os.path.join(OUTPUT_DIR, "top4_feature_importance.png"))
    for _, row in top4.iterrows():
        f_raw = row["Feature"]
        f_label = row["Feature_pretty"]
        outp = os.path.join(OUTPUT_DIR, f"compare_{slug(f_raw)}.png")
        plot_group_feature(df, f_raw, f_label, outp)

    save_group_accuracy(df, os.path.join(OUTPUT_DIR, "group_accuracy.png"))

    print("\n=== Model & Outputs ===")
    print(f"Chosen model: {best_name} | CV (tuned, train) mean: {best_cv_mean if not np.isnan(best_cv_mean) else 'n/a'} (splits={used_splits})")
    if not np.isnan(cv_full_mean):
        print(f"Repeated CV (full dataset) mean: {cv_full_mean:.3f} ± {cv_full_std:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(top4[["Feature_pretty","Importance"]].to_string(index=False))
    print(f"\nFiles in: {OUTPUT_DIR}")
    print(" - top4_feature_importance.png")
    print(" - group_accuracy.png")
    for f in top4["Feature"]:
        print(f" - compare_{slug(f)}.png")
    print(" - top4_features.csv")
    print(" - group_accuracy.csv")
    print(" - player_accuracy.csv")
    print(" - playtest_features.csv")
    if test_acc < ACC_TARGET:
        print(f"\n[NOTE] Test accuracy {test_acc:.2%} is below 80%. With ~10 samples, a single split is unstable. Use the repeated CV mean as the headline.")
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "vhb_rf_model.joblib"))
    print("[OK] Model saved → outputs\\vhb_rf_model.joblib")

if __name__ == "__main__":
    main()
