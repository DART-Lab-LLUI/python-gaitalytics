""" 
this script is used to only extract and compute the CoM signals,
from an existing github repository within DART(LLUI)
"""
from pathlib import Path
import sys
import pandas as pd
from matplotlib import pyplot as plt
events_branch = Path(r"..\RepoCode\events\python-gaitalytics").resolve()
pbt_branch    = Path(r"..\RepoCode\pbt-analysis").resolve()
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
    def __init__(self, c3d_path: Path, config, model_com_body):
        self.c3d_path       = c3d_path
        self.config         = config
        self.model_com_body = model_com_body
        self.trial          = None
        self.events         = None

    def compute(self):
        
        # load the trial and compute the CoM signals employing 
        # model_com_body from cloned repo
        self.trial = load_c3d_trial(self.c3d_path, self.config)
        self.trial = self.model_com_body(self.trial)

        # detect events in trial and extract a .csv for check
        if self.trial.events is not None:
            print(f"Events already detected in {self.c3d_path.name}. Skipping event detection.")
            print("event", self.trial.events.columns.tolist())
            print("\nFirst few events:")
            print(self.trial.events.head())
            self.trial.events.to_csv(self.c3d_path.parent / f"events.csv", index=False)
            print("Events saved to CSV format.")
            return self.trial
        else:
            print(f"No events detected in {self.c3d_path.name}.")
            return self.trial
        
        
    def export_trial(self, out_dir: Path):

        # export trial , .nc file for markers and events is automatically saved
        out_dir.mkdir(parents=True, exist_ok=True)
        export_trial(self.trial, self.c3d_path.parent)
        print(f"[UNSEG] Exported {self.c3d_path.name} → {self.c3d_path.parent}")


        
if __name__ == "__main__":
    from gaitalytics import api
    import xarray as xr
    data_root       = Path("../../data/PBT/Young/Cereneo_SR_16/Pre.4/repair")
    # data_root       = Path("../../data/PBT/Elderly/38")
    # data_root       = Path("../../data/CGA/51/FWS")
    config_path     = Path("pig_config.yaml")

    config = api.load_config(config_path)

    for c3d in sorted(data_root.rglob("*.c3d")):
        rel   = c3d.parent.relative_to(data_root)
        markers_nc = data_root / 'markers.nc'
        tp = TrialProcess(c3d, config, model_com_body)
        tp.compute()
        tp.export_trial(rel)
      

        

    
