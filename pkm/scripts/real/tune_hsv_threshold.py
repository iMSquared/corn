#!/usr/bin/env python3

import copy
import numpy as np
import cv2
import time
import pickle
import json
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm

from pkm.real.rs_camera import RSCamera
from rt_cfg import RuntimeConfig
from pkm.real.seg.segmenter import rgb2hsv
import torch as th

def setup_gui(bounds, values, on_bound = None, name='hsv-gui'):
    cv2.namedWindow(name)

    b = copy.deepcopy(bounds)

    def on_pos(value, key:str, index:int):
        b[key][index]=value
        if on_bound is not None:
            on_bound(b)

    for k, (b_min, b_max) in bounds.items():
        count = (b_max - b_min)
        # assert count <= 255
        vmin, vmax = values[F'{k}']
        cv2.createTrackbar(F'{k}_min', name, vmin, count,
                           partial(on_pos, key=k, index=0))
        cv2.createTrackbar(F'{k}_max', name, vmax, count,
                           partial(on_pos, key=k, index=1))

def tune_camera(rt_cfg, device_id: str):
    bounds = {
        'h1' : [0, 180],
        'h2' : [0, 180],
        's' : [0, 255],
        'v' : [0, 255]
    }

    values = copy.deepcopy(bounds)

    if Path(rt_cfg.table_color_file).exists():
        with open(rt_cfg.table_color_file, 'rb') as fp:
            values = pickle.load(fp)[device_id]

    cams = {
        device_id : RSCamera(RSCamera.Config(
            device_id=device_id,
            fps=60,
            poll=False))
            }
    
    imgs = {k:cam.get_images()
        for (k, cam) in cams.items()}
    time.sleep(1)

    def on_bound(b, k):
        rgb = imgs[k]['color']
        # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv = rgb2hsv(th.as_tensor(rgb)/255.0).detach().cpu().numpy()
        hsv[..., 0] *= 0.5
        hsv[..., 1] *= 255
        hsv[..., 2] *= 255
        hsv = hsv.astype(np.uint8)

        bounds.update(b)

        # cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV, dst = hsv)

        keys_1 = ['h1', 's', 'v']
        keys_2 = ['h2', 's', 'v']

        lo_1 = [b[k][0] for k in keys_1]
        hi_1 = [b[k][1] for k in keys_1]
        lo_2 = [b[k][0] for k in keys_2]
        hi_2 = [b[k][1] for k in keys_2]
        
        lo_1 = tuple(lo_1)
        hi_1 = tuple(hi_1)
        lo_2 = tuple(lo_2)
        hi_2 = tuple(hi_2)

        msk_1 = cv2.inRange(hsv, lo_1, hi_1)
        msk_2 = cv2.inRange(hsv, lo_2, hi_2)
        # msk   = np.logical_or(msk_1, msk_2)
        msk = (msk_1 | msk_2)

        cv2.namedWindow(F'{k}-msk', cv2.WINDOW_NORMAL)
        cv2.imshow(F'{k}-msk', msk)

    def call_on_bounds(*args, **kwds):
        for k in cams.keys():
            on_bound(*args, **kwds, k = k)

    # for k in cams.keys():
    setup_gui(bounds, values, on_bound=call_on_bounds,
                name='hsv-gui')

    try:
        while True:
            for k, cam in cams.items():
                imgs[k] = cam.get_images()    
                cv2.imshow(F'{k}-rgb', cv2.cvtColor(imgs[k]['color'], cv2.COLOR_RGB2BGR))

            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        return bounds



def main():
    rt_cfg = RuntimeConfig()
    
    table_colors = {}
    for dev_id in tqdm(rt_cfg.cam_ids):
        table_colors[dev_id] = tune_camera(rt_cfg, dev_id)
    with open(rt_cfg.table_color_file, 'wb') as fp:
        pickle.dump(table_colors, fp)


if __name__ == '__main__':
    main()  