#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
from matplotlib import pyplot as plt
events_branch = Path(
    r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\RepoCode\events\python-gaitalytics"
).resolve()
pbt_branch    = Path(
    r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\RepoCode\pbt-analysis"
).resolve()
sys.path.insert(0, str(events_branch))
sys.path.insert(0, str(pbt_branch))

from gaitalytics.api import (
    export_trial,
    load_config,
    load_c3d_trial,
    get_event_detector,
    detect_events,
    check_events,
    find_optimal_detectors,
    write_events_to_c3d,
)

from src.utils.modelling.model import model_com_body

class TrialProcess:
    """
    Processes a single C3D trial:
      - Computes CoM + markers
      - Bootstraps with Zen or Desailly
      - Trims to first N cycles
      - Auto‐selects among Zen/Des/AC1/AC6
      - Detects final events
      - Exports events to CSV
    """
    def __init__(self, c3d_path: Path, config, model_com_body):
        self.c3d_path       = c3d_path
        self.config         = config
        self.model_com_body = model_com_body
        self.trial          = None
        self.events         = None

    def compute(self):
        """Load raw C3D and compute CoM."""
        self.trial = load_c3d_trial(self.c3d_path, self.config)
        self.trial = self.model_com_body(self.trial)
       
        return self.trial
    
    def export_trial(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        export_trial(self.trial, self.c3d_path.parent)

        print(f"[UNSEG] Exported {self.c3d_path.name} → {self.c3d_path.parent}")

    @staticmethod
    def _trim_to_n_cycles(df: pd.DataFrame, n_cycles: int = 15) -> pd.DataFrame:
        """
        Keep only the first N gait cycles (HS→HS) per leg.
        """
        out = []
        for side in df['context'].unique():
            leg     = df[df.context == side].sort_values('time')
            strikes = leg[leg.label == 'Foot Strike']
            if len(strikes) < n_cycles + 1:
                raise ValueError(
                    f"Found only {len(strikes)} strikes for {side}, need {n_cycles+1}"
                )
            cutoff = strikes.iloc[n_cycles].time
            out.append(leg[leg.time <= cutoff])
        return pd.concat(out).sort_values(['time','context']).reset_index(drop=True)

    def auto_tune_events(self, n_cycles: int = 15):
        """
        1) Bootstrap with Zen (HS→TO) or if that fails, Desailly
        2) Trim that to first N cycles
        3) Auto‐select best among Zen, Des, AC1, AC6
        4) Final detect & check
        """
   
        for bootstrap_method in ["Zen", "Des"]:
            det = get_event_detector(bootstrap_method, bootstrap_method, self.config)
            cand = detect_events(self.trial, det)
            try:
                check_events(cand)
                print(f"✅ Bootstrap {bootstrap_method} OK for {self.c3d_path.name}")
                bootstrap = cand
                break
            except ValueError:
                print(f"⚠️  Bootstrap {bootstrap_method} failed, trying next…")
        else:
            print("❌ Both Zen and Des bootstrap failed; using raw Zen anyway")
            det = get_event_detector("Zen", "Zen", self.config)
            bootstrap = detect_events(self.trial, det)


        ref = self._trim_to_n_cycles(bootstrap, n_cycles=n_cycles)
        self.trial.events = ref

        detector, feedback = find_optimal_detectors(
            self.trial,
            self.config,
            method_list=["Zen", "Des", "AC1", "AC6"]
        )
        print(f"🔧 Auto‐selector feedback for {self.c3d_path.name}:", feedback)

        final = detect_events(self.trial, detector)
        try:
            check_events(final)
            print(f"✅ Final events OK for {self.c3d_path.name}")
        except ValueError as e:
            print(f"⚠️ Final sequence error for {self.c3d_path.name}: {e}")

        self.events = final
        self.trial.events = final
        return final

    def export_events(self, out_dir: Path):
        """Export self.events to CSV."""
        if self.events is None:
            raise RuntimeError("No events to export; run auto_tune_events() first")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv = out_dir / f"{self.c3d_path.stem}_events.csv"
        self.events.to_csv(csv, index=False)
        print(f"[EVENTS] {self.c3d_path.name} → {csv}")

        
if __name__ == "__main__":
    from gaitalytics import api
    import xarray as xr
    data_root       = Path("../../data/PBT/Young/Cereneo_SR_20/Pre")
    config_path     = Path("pig_config.yaml")
    out_events_root = data_root / "markers+Com"

    config = api.load_config(config_path)

    for c3d in sorted(data_root.rglob("*.c3d")):
        rel   = c3d.parent.relative_to(data_root)
        evdir = out_events_root / rel
        markers_nc = data_root / 'markers.nc'

        tp = TrialProcess(c3d, config, model_com_body)
        tp.compute()
      

        tp.export_trial(rel)
        tp.auto_tune_events(n_cycles=10)
        tp.export_events(evdir)

    
