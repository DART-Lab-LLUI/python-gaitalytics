from gaitalytics import api
from gaitalytics import utils


def main():
    settings_file = "settings/hbm_pig.yaml"
    file_path = "./tests/data/Baseline.4.c3d"
    out_path = "./tests/data/"

    # load configs
    configs = utils.ConfigProvider(settings_file)

    api.model_data(file_path, out_path, configs, methode=api.MODELLING_CMOS, belt_speed=0.8, dominant_leg_length=998)


if __name__ == "__main__":
    main()
