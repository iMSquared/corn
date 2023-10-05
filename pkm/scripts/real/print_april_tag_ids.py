#!/usr/bin/env python3

from pkm.real.rs_camera import RSCamera
from dt_apriltags import Detector
from rt_cfg import RuntimeConfig
import pickle
import cv2
import numpy as np

from hydra_zen import (store, zen)


@store(name='print_tag_ids')
def _main(object: str,
          tag_size: float = 0.015):

    rt_cfg = RuntimeConfig()
    at_detector = Detector(families='tag36h11')
    camera = RSCamera(RSCamera.Config(rt_cfg.table_cam_device_id,
                                      poll=False))

    seen_ids = set()
    try:
        while True:
            imgs = camera.get_images()
            if imgs is None:
                continue

            # gray = rgb2gray(rgb)
            gray = cv2.cvtColor(imgs['color'],
                                cv2.COLOR_RGB2GRAY)

            tags = at_detector.detect(gray.astype(np.uint8), True,
                                      camera.cam_param, tag_size)
            tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t)
                         for tag in tags}

            new_seen_ids = seen_ids.union(tags_dict.keys())
            if len(new_seen_ids) > len(seen_ids):
                print(list(new_seen_ids))
            seen_ids = new_seen_ids
    finally:
        with open(F'/tmp/{object}_tag_ids.pkl', 'wb') as fp:
            pickle.dump(seen_ids, fp)


def main():
    store.add_to_hydra_store()
    zen(_main).hydra_main(config_name='print_tag_ids',
                          version_base='1.1',
                          config_path=None)


if __name__ == '__main__':
    main()
