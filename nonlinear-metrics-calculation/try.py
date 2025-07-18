#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------
# 1) Make sure your local branch shadows any pip install of gaitalytics
# -------------------------------------------------------------------
branch_root = Path(r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\RepoCode\events\python-gaitalytics").resolve()
sys.path.insert(0, str(branch_root))

# -------------------------------------------------------------------
# 2) Import exactly what we need from your branch’s API
# -------------------------------------------------------------------
from gaitalytics.api import (
    load_config,
    load_c3d_trial,
    get_event_detector,
    get_ref_from_GRF,
    detect_events,
    check_events,
    find_optimal_detectors,
    write_events_to_c3d,
)

# -------------------------------------------------------------------
# 3) Helper: keep only the first N gait cycles (HS→HS) per leg
# -------------------------------------------------------------------
def trim_to_n_cycles(df: pd.DataFrame, n_cycles: int = 10) -> pd.DataFrame:
    out = []
    for side in df['context'].unique():
        leg = df[df.context == side].sort_values('time')
        strikes = leg[leg.label == 'Foot Strike']
        if len(strikes) < n_cycles + 1:
            raise ValueError(f"Found only {len(strikes)} strikes for {side}, need {n_cycles+1}")
        cutoff = strikes.iloc[n_cycles].time
        out.append(leg[leg.time <= cutoff])
    return pd.concat(out).sort_values(['time','context']).reset_index(drop=True)

# -------------------------------------------------------------------
# 4) Main pipeline
# -------------------------------------------------------------------
def main():
    config = load_config("pig_config.yaml")
    trial  = load_c3d_trial(Path("../../data/CGA/03/5050602_20231230_SWS.3.c3d"), config)

    zen       = get_event_detector("Zen","Zen",config)
    bootstrap = detect_events(trial, zen)
    try: check_events(bootstrap)
    except ValueError: print("⚠️ Zeni bootstrap didn’t alternate, proceeding")

    ref10     = trim_to_n_cycles(bootstrap, n_cycles=15)
    trial.events = ref10

    # AUTO‐SELECT among the four kinematic methods
    detector, feedback = find_optimal_detectors(
        trial,
        config,
        method_list=["Zen","Des","AC1","AC6"]
    )
    print("🔧 Auto‐selector feedback:", feedback)

    final = detect_events(trial, detector)
    try: check_events(final)
    except ValueError as e: print("❌ Final check failed:", e)

    final.to_csv("../../data/CGA/03/5050602_20231230_NEW.csv", index=False)
    print("💾 Done.")


if __name__ == "__main__":
    main()
