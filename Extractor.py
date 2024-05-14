###########################################################################
#                                                                         #
# Source: https://github.com/chi0tzp/PyVideoFramesExtractor/tree/master ###
#                                                                         #
###########################################################################

import os
import os.path as osp
import cv2
import math
from multiprocessing import Pool

# Supported video and frame extensions
supported_video_ext = ('.avi', '.mp4')
supported_frame_ext = ('.jpg', '.png')


class FrameExtractor:
    """Extract frames from video file and save them under a given output directory.

        Args:
            video_file (str)  : input video filename
            output_dir (str)  : output directory where video frames will be extracted
            frame_ext (str)   : extracted frame file format
            sampling (int)    : sampling rate -- extract one frame every given number of seconds.
                                Default=-1 for extracting all available frames
    """
    def __init__(self, video_file, output_dir, frame_ext='.jpg', sampling=-1):
        if not osp.exists(video_file):
            raise FileExistsError('Video file {} does not exist.'.format(video_file))
        self.video_file = video_file
        self.output_dir = output_dir
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if frame_ext not in supported_frame_ext:
            raise ValueError("Not supported frame file format: {}".format(frame_ext))
        self.frame_ext = frame_ext
        self.sampling = sampling
        self.video = cv2.VideoCapture(self.video_file)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sampling != -1:
            self.video_length = self.video_length // self.sampling

    def extract(self):
        success, frame = self.video.read()
        frame_cnt = 0
        while success:
            curr_frame_filename = osp.join(self.output_dir, "{:08d}{}".format(frame_cnt, self.frame_ext))
            cv2.imwrite(curr_frame_filename, frame)
            success, frame = self.video.read()
            if self.sampling != -1:
                frame_cnt += math.ceil(self.sampling * self.video_fps)
                self.video.set(1, frame_cnt)
            else:
                frame_cnt += 1


def extract_video_frames(video_file, output_root, sampling):
    video_basename = osp.basename(video_file).split('.')[0]
    video_ext = osp.splitext(video_file)[-1]
    if video_ext not in supported_video_ext:
        raise ValueError("Not supported video file format: {}".format(video_ext))
    output_dir = osp.join(output_root, '{}_frames'.format(video_basename))
    if os.stat(video_file).st_size > 0:
        if not osp.isdir(output_dir):
            # Set up video extractor for given video file
            extractor = FrameExtractor(video_file=video_file, output_dir=output_dir, sampling=sampling)
            # Extract frames
            extractor.extract()
    else:
        os.remove(video_file)


def extract_frames_from_videos(video=None, dir=None, sampling=-1, output_root='extracted_frames', workers=None):
    if video:
        video_basename = osp.basename(video).split('.')[0]
        video_ext = osp.splitext(video)[-1]
        if video_ext not in supported_video_ext:
            raise ValueError("Not supported video file format: {}".format(video_ext))
        output_dir = osp.join(output_root, '{}_frames'.format(video_basename))
        extractor = FrameExtractor(video_file=video, output_dir=output_dir, sampling=sampling)
        extractor.extract()

    if dir:
        video_list = []
        for r, d, f in os.walk(dir):
            for file in f:
                file_basename = osp.basename(file).split('.')[0]
                file_ext = osp.splitext(file)[-1]
                if file_ext in supported_video_ext:
                    video_list.append((osp.join(r, file), osp.join(output_root, osp.relpath(r, dir), "{}_frames".format(file_basename))))

        with Pool(processes=workers) as p:
            p.starmap(extract_video_frames, [(v[0], v[1], sampling) for v in video_list])

# Example usage commented out to prevent execution
# extract_frames_from_videos(dir="path_to_videos", sampling=2, output_root="extracted_frames_folder", workers=4)
