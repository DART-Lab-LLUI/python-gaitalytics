from pathlib import Path

from gaitalytics import api
from gaitalytics import file
from gaitalytics import utils


def main():
    settings_file = "settings/hbm_pig.yaml"
    buffered_path = Path("./out/processed.h5")
    out_path = Path("./out/analysis.h5")

    configs = utils.ConfigProvider(settings_file)

    loaded_cycles = api.extract_cycles_buffered(buffered_path, configs)

    results = api.analyse_data(loaded_cycles, configs)

    storage = file.Hdf5FileStore(out_path, configs)
    storage.save_analysis(results)


if __name__ == "__main__":
    main()
