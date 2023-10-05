#!/usr/bin/env bash

python3 collect_calibration_data.py
python3 calibrate_transforms.py
python3 multi_cam_calibration.py
