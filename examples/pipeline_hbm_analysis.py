from pathlib import Path

from gaitalytics import api
from gaitalytics import utils
from gaitalytics import model


def main():
    settings_file = "settings/hbm_pig.yaml"
    buffered_path = "./out/processed.h5"
    out_path = Path("./spatio_temp")

    configs = utils.ConfigProvider(settings_file)

    loaded_cycles = api.extract_cycles_buffered(buffered_path, configs)

    results = api.analyse_data(loaded_cycles, configs)

    if not out_path.exists():
        out_path.mkdir()
    results.to_csv(f"{out_path}/spatio_temp.csv")


if __name__ == "__main__":
    main()
