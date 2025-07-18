
from pathlib import Path
import sys

# Configure private branch for CoM computation
package_dir = Path(r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\RepoCode\pbt-analysis").resolve()
sys.path.append(str(package_dir))


from gaitalytics import api
import pandas as pd



class TrialProcess:
    """
    Encapsulates processing steps for a single C3D trial file, up to event detection.

    Attributes:
        c3d_path (Path): Path to the input .c3d file.
        config: Loaded YAML configuration for gaitalytics API.
        model_com_body: Function to compute center of mass on the trial.
        trial: In-memory trial object after CoM computed.
        events: Detected gait events attached to the trial.
    """
    def __init__(self, c3d_path: Path, config, model_com_body):
        self.c3d_path = c3d_path
        self.config = config
        self.model_com_body = model_com_body
        self.trial = None
        self.events = None

    def compute(self):
        """
        Load raw C3D and compute CoM.
        """
        self.trial = api.load_c3d_trial(self.c3d_path, self.config)
        self.trial = self.model_com_body(self.trial)
        return self.trial

    def export_unsegmented(self, out_dir: Path):
        """
        Export the computed trial (markers + CoM) to out_dir.
        """
        if self.trial is None:
            raise RuntimeError("Cannot export: compute() not called yet.")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / self.c3d_path.stem
        api.export_trial(self.trial, out_base)
        print(f"[UNSEG] Exported {self.c3d_path.name} → {out_base}.*")



    def get_events(self):
        """
        Detect gait events and attach to self.trial.
        """
        if self.trial is None:
            self.compute()
        self.events = api.detect_events(self.trial, self.config)
        try:
            api.check_events(self.events)
        except ValueError as e:
            print(f"⚠️ Event check failed for {self.c3d_path.name}: {e}")
        self.trial.events = self.events
        return self.events

    def export_events(self, out_dir: Path):
        """
        Export the detected events table to CSV in out_dir.
        """
        if self.events is None:
            raise RuntimeError("Cannot export events: get_events() not called yet.")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{self.c3d_path.stem}_events.csv"
        self.events.to_csv(csv_path)
        print(f"[EVENTS] Exported {self.c3d_path.name} → {csv_path}")


if __name__ == '__main__':
    # 1) Define paths
    from pathlib import Path
    data_root       = Path('../../data/PBT/Young/Cereneo_SR_17')
    config_path     = Path('pig_config.yaml')
    out_unseg_root  = Path(data_root, 'markers+Com')
    out_events_root = Path(data_root,'markers+Com/simple')

    # 2) Load YAML config
    config = api.load_config(config_path)
    # 3) Import CoM model (external branch or fallback)
    try:
        from src.utils.modelling.model import model_com_body
    except ImportError:
        from gaitalytics import model_com_body

    # 4) Loop through all C3D files and process
    for c3d_file in data_root.rglob('*.c3d'):
        rel = c3d_file.parent.relative_to(data_root)
        unseg_dir  = out_unseg_root  / rel
        events_dir = out_events_root / rel

        tp = TrialProcess(c3d_file, config, model_com_body)
        tp.compute()
        tp.export_unsegmented(unseg_dir)
        tp.get_events()
        tp.export_events(events_dir)
