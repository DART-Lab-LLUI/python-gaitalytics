from gaitalytics import api

config = api.load_config("./tests/pig_config.yaml")
trial = api.load_c3d_trial("./tests/test_small.c3d", config)
trial_segmented = api.segment_trial(trial)
