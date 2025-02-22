from gaitalytics import api

config = api.load_config("./tests/pig_config.yaml")
trial = api.load_c3d_trial("./tests/test_small.c3d", config)

trial_ref = api.get_ref_from_GRF(trial, config)
event_detector, user_show = api.find_optimal_detectors(trial_ref, config, method_list=["Zen", "Des", "AC1", "AC6"])
events = api.detect_events(trial, event_detector)
print(user_show)

try:
    api.check_events(trial.events)
except ValueError as e:
    print(e)

api.write_events_to_c3d("./tests/test_small.c3d", events,
                        "./out/test_small_events.c3d")