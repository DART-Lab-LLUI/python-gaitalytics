from gaitalytics import api

config = api.load_config("./tests/pig_config.yaml")
trial = api.load_c3d_trial("./tests/test_small.c3d", config)
events = api.detect_events(trial, config)

try:
    api.check_events(trial.events)
except ValueError as e:
    print(e)

api.write_events_to_c3d("./tests/test_small.c3d", events,
                        "./out/test_small_events.c3d")

