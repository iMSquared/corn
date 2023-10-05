    #!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
import cv2
import pyrealsense2 as rs


class RSCamera:
    @dataclass
    class Config:
        device_id: str = '233622074125'
        img_width: int = 640
        img_height: int = 480
        fps: float = 30
        debug: bool = False
        cloud_size: int = 512
        depth_scale: float = 1000.0
        poll:bool = True

    def __init__(self, cfg: Config, ctx=None,
                 is_master: bool = True):
        self.cfg = cfg

        # if ctx is not None:
        #     mode = 1 if is_master else 2
        #     for device in ctx.query_devices():
        #         print(device.get_info(rs.camera_info.serial_number))
        #         if device.get_info(rs.camera_info.serial_number) == cfg.device_id:
        #             print(device.first_depth_sensor().set_option(
        #                 rs.option.inter_cam_sync_mode, mode))
        if ctx is not None:
            self.pipeline = rs.pipeline(ctx)
        else:
            self.pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(cfg.device_id)
        rs_cfg.enable_stream(rs.stream.depth, cfg.img_width,
                             cfg.img_height, rs.format.z16, cfg.fps)
        rs_cfg.enable_stream(rs.stream.color, cfg.img_width,
                             cfg.img_height, rs.format.rgb8, cfg.fps)

        if ctx is not None:
            # s = self.pipe_prof.get_device().query_sensors()[0]
            for i in range(len(ctx.devices)):
                sn = ctx.devices[i].get_info(rs.camera_info.serial_number)
                print(sn)
                if sn == cfg.device_id:
                    device_idx = i
                    # device.first_depth_sensor()
            s = ctx.devices[device_idx].first_depth_sensor()
            mode = 1 if is_master else 2
            s.set_option(rs.option.global_time_enabled, 1)
            s.set_option(rs.option.inter_cam_sync_mode, mode)
            s.set_option(rs.option.output_trigger_enabled, 1)
            s.set_option(rs.option.frames_queue_size, 2)

        if cfg.poll:
            self.queue = rs.frame_queue(2)
            self.pipe_prof = self.pipeline.start(rs_cfg, self.queue)
        else:
            self.pipe_prof = self.pipeline.start(rs_cfg)

        # s = self.pipe_prof.get_device().query_sensors()[0]
        # print('???', cfg.device_id, s.get_option(rs.option.inter_cam_sync_mode))

        s = self.pipe_prof.get_device().query_sensors()[1]
        # s.set_option(rs.option.exposure, 120)
        s.set_option(rs.option.enable_auto_exposure, 1)

        # if ctx is not None:
        #     s = self.pipe_prof.get_device().query_sensors()[0]
        #     mode = 1 if is_master else 2
        #     s.set_option(rs.option.inter_cam_sync_mode, mode)
        color_stream = self.pipe_prof.get_stream(rs.stream.color)

        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print('intrinsics', intrinsics)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = intrinsics
        self.K = np.asarray([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]])
        self.cam_param = \
            (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    def poll_images(self):
        if not self.cfg.poll:
            raise ValueError('polling disabled for now.')
        # frames = self.pipeline.poll_for_frames()
        frames = self.queue.poll_for_frame().as_frameset()
        #.get_depth_frame()
        # print(frames.size())
        return frames

    def proc_images(self, frames):
        cfg = self.cfg
        # print('frames', frames)
        print(self.cfg.device_id, frames.get_depth_frame().get_timestamp())

        frames = self.align.process(frames)

        depth_image = np.array(
            frames.get_depth_frame().get_data()) / cfg.depth_scale
        color_image = np.asanyarray(
            frames.get_color_frame().get_data())
        return {
            'depth': depth_image,
            'color': color_image
        }

    def get_images(self, poll:bool = False):
        cfg = self.cfg
        poll = cfg.poll
        if poll:
            frames = self.poll_images()
            if frames is None or frames.size() <= 0:
                return None
        else:
            frames = self.pipeline.wait_for_frames()

        frames = self.align.process(frames)
        # print(self.cfg.device_id, frames.get_depth_frame().get_timestamp())

        depth_image = np.array(
            frames.get_depth_frame().get_data()) / cfg.depth_scale
        # color_image = None
        color_image = np.asanyarray(
            frames.get_color_frame().get_data())
        return {
            'depth': depth_image,
            'color': color_image
        }


def main():
    camera = RSCamera(RSCamera.Config())
    print(camera.K)

    for _ in range(128):
        images = camera.get_images()

        cv2.imshow('color', images['color'])
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
