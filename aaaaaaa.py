#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
from models.utils_dataloader import LoadFocalFolder
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
from torch.utils.data import DataLoader
torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--GPU_ID', type=str, default='2',
        help='GPU number')
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[3840, 2160],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    device = torch.device("cuda:%s" % opt.GPU_ID if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU %s' % opt.GPU_ID)
    else:
        print('Using CPU')
    print('Running inference on device \"{}\"'.format(device))

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    # vs = VideoStreamer(opt.input, opt.resize, opt.skip,
    #                    opt.image_glob, opt.max_length)
    # frame, ret = vs.next_frame()
    # assert ret, 'Error when reading the first frame (try different --input?)'
    #
    # frame_tensor = frame2tensor(frame, device)
    # last_data = matching.superpoint({'image': frame_tensor})
    # last_data = {k+'0': last_data[k] for k in keys}
    # last_data['image0'] = frame_tensor
    # last_frame = frame
    # last_image_id = 0

    def get_last_data(frame):
        frame_tensor = frame2tensor(frame, device)
        last_data = matching.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor
        return last_data

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    # if not opt.no_display:
    #     cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    # else:
    #     print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    dataset = LoadFocalFolder(opt.input, frame_range=[3, 65], focal_range=None)
    H_list = []
    # for idx, (last_frame, frames) in enumerate(train_loader):
    # with torch.no_grad():
    #     for idx in range(dataset.__len__()):
    #         if idx == 0:
    #             last_frame, frames = dataset.__getitem__(idx)
    #             print('last_frame',last_frame.shape)
    #             for frame_idx, frame in enumerate(frames):
    #
    #             # while True:
    #             #     frame, ret = vs.next_frame()
    #             #     if not ret:
    #             #         print('Finished demo_superglue.py')
    #             #         break

    # timer.update('data')
    # # stem0, stem1 = last_image_id, vs.i - 1
    # stem0, stem1 = 7, frame_idx
    last_frame = '/media/mnt/dataset/2/images/004.png'
    last_data = get_last_data(last_frame)

    frame_tensor = frame2tensor(frame, device)
    pred = matching({**last_data, 'image1': frame_tensor})

    kpts0 = last_data['keypoints0'][0].cpu().numpy() # 기준카메라
    desc0 = last_data['descriptors0'][0].cpu().numpy()
    desc1 = pred['descriptors1'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy() # 탐색 카메라
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()


    # print('matches:', matches)
    valid = matches > -1
    # valid = []
    # for i, match in enumerate(matches):
    #     if 250 < match < 350 and i % 3 == 0:
    #         valid.append(True)
    #     elif 100 < match < 200 and i % 3 == 0:
    #         valid.append(True)
    #     else:
    #         valid.append(False)
        # valid.append(250 < match < 350 and i % 3 == 0)
    # valid = 250 < matches and matches < 350
    # mkpts0 = kpts0[valid]
    # mkpts1 = kpts1[matches[valid]]
    matcher = cv2.BFMatcher_create()
    desc0 = np.transpose(desc0)
    desc1 = np.transpose(desc1)
    matches = matcher.match(desc0, desc1)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:80]
    pts1 = np.array([kpts0[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([kpts1[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    H_list.append(H)
np.save('H_result_2.npy', H_list)


