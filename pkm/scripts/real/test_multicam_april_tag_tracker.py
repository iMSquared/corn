#!/usr/bin/env python3

from pkm.real.track.multicam_april_tag_tracker import MulticamAprilTagTracker
from rt_cfg import RuntimeConfig
from pkm.real.rs_camera import RSCamera
import numpy as np

from hydra_zen import (store, zen, hydrated_dataclass)


@store(name='test_april')
def _main(object: str, tag_size: float = 0.015):
    rt_cfg = RuntimeConfig()
    cameras = {
        device_id: RSCamera(RSCamera.Config(poll=False,
                                            device_id=device_id))
        for device_id in rt_cfg.cam_ids
    }
    intrinsics = [cameras[device_id].K for device_id in rt_cfg.cam_ids]
    tracker = MulticamAprilTagTracker(rt_cfg=rt_cfg,
                                      debug=True,
                                      tag_size=tag_size,
                                      #   object='ceramic-cup',
                                      # april_offset_file=rt_cfg.april_offset_file('chapa-bottle'),
                                      april_offset_file=rt_cfg.april_offset_file(
                                          object),
                                      # object='dispenser',
                                      extrinsics=rt_cfg.extrinsics,
                                      intrinsics=intrinsics,

                                      max_angle_between_frames=float('inf')
                                      # max_angle_between_frames=np.deg2rad(30),
                                      )
    for _ in range(10000):
        color_images = [
            cameras[device_id].get_images()['color']
            for device_id in rt_cfg.cam_ids]
        depth_images = [
            cameras[device_id].get_images()['depth']
            for device_id in rt_cfg.cam_ids]

        tracker(color_images, depth_images=depth_images)


def main():
    store.add_to_hydra_store()
    zen(_main).hydra_main(config_name='test_april',
                          version_base='1.1',
                          config_path=None)


if __name__ == '__main__':
    main()
