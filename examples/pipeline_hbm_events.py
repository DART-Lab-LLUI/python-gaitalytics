from gaitalytics import api
from gaitalytics import utils
from gaitalytics import c3d_reader


def main():
    settings_file = "settings/hbm_pig.yaml"
    file_path = "./tests/data/Baseline.3.c3d"
    out_path = "./tests/data/"

    # load configs
    configs = utils.ConfigProvider(settings_file)

    api.detect_gait_events(file_path, out_path, configs, show_plot=False, file_handler_class=c3d_reader.BtkFileHandler)


if __name__ == "__main__":
    main()
