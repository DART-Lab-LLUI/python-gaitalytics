from pathlib import Path

from gaitalytics import api
from gaitalytics import utils
from gaitalytics import model


def main():
    settings_file = "settings/hbm_pig.yaml"
    file_path = "./tests/data/Baseline.5.c3d"
    buffered_path_raw = Path("./out/processed.h5")
    buffered_path_norm = Path("./out/processed.h5")

    configs = utils.ConfigProvider(settings_file)
    if not buffered_path_raw.exists():
        buffered_path_raw.parent.mkdir(parents=True, exist_ok=True)
        cycle_data = api.extract_cycles(file_path, configs, buffer_output_path=buffered_path_raw)
    else:
        cycle_data = api.extract_cycles_buffered(buffered_path_raw, configs)[model.ExtractedCycleDataCondition.RAW_DATA]

    api.normalise_cycles(configs, cycle_data, buffer_output_path=buffered_path_norm)


# __name__
if __name__ == "__main__":
    main()
