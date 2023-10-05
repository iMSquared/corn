# Scripts for real-robot deployment

---

## Setup

Calibrate the cameras and robot extrinsics:
```
$ python3 collect_calibration_data.py # -> ${PKM_RUNTIME_DIR}/pq_list.pkl
$ python3 calibrate_transforms.py # -> ${PKM_RUNTIME_DIR}/transforms.pkl
$ python3 multi_cam_calibration.py # -> ${PKM_RUNTIME_DIR}/extrinsics.pkl
```

## Main

### Generate current task configuration

```
$ python3 sample_goal_pose.py # -> ${PKM_RUNTIME_DIR}/task{index:04d}.pkl
```

### Run the actual policy and collect runtime logs

Requires the following files:

* `${PKM_RUNTIME_DIR}/extrinsics.pkl`
* `${PKM_RUNTIME_DIR}/task*.pkl`

By default, the latest transforms/extrinsics/task files are selected.

```
$ python3 controller.py
```

## Tool

Control Franka with Joystick:
```
$ python3 joystick_controller.py
```

### Replay logs

```
$ python3 replay_log.py path=...
```

### Test perception stack

Run the perception stack independently, without running controller:

```
$ python3 test_perception.py
```
