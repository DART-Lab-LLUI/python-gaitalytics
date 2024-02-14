from gaitalytics import api
from gaitalytics import utils


def main():
    settings_file = "settings/hbm_pig.yaml"
    file_path = "./tests/data/Baseline.3.c3d"
    out_path = "./tests/data/"

    # load configs
    configs = utils.ConfigProvider(settings_file)

    api.detect_gait_events(file_path, out_path, configs, show_plot=False)


if __name__ == "__main__":
    main()
