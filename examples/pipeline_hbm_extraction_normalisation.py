from pathlib import Path

from gaitalytics import api
from gaitalytics import utils


def main():
    settings_file = "settings/hbm_pig.yaml"
    file_path = "./tests/data/Baseline.5.c3d"
    buffered_path_raw = Path("./raw")
    buffered_path_norm = Path("./norm")

    configs = utils.ConfigProvider(settings_file)
    if not buffered_path_raw.exists():
        buffered_path_raw.mkdir(parents=True, exist_ok=True)
        cycle_data = api.extract_cycles(file_path, configs, buffer_output_path=buffered_path_raw)
    else:
        cycle_data = api.extract_cycles_buffered(buffered_path_raw, configs).get_raw_cycle_points()

    buffered_path_norm.mkdir(parents=True, exist_ok=True)
    api.normalise_cycles(file_path, cycle_data, buffer_output_path=buffered_path_norm)


# __name__
if __name__ == "__main__":
    main()
