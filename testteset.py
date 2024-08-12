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
from copy import deepcopy
import math

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
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


def get_last_data(frame):
    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    return last_data


def perspective(img1, img2, color, get_last_data):
    last_data = get_last_data

    frame_tensor = frame2tensor(img2, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()  # 기준카메라
    kpts1 = pred['keypoints1'][0].cpu().numpy()  # 탐색 카메라
    matches0 = pred['matches0'][0].cpu().numpy()
    matching_scores = pred['matching_scores0'][0].detach().cpu().numpy()
    desc0 = last_data['descriptors0'][0].detach().cpu().numpy()
    desc1 = pred['descriptors1'][0].detach().cpu().numpy()
    valid = matches0 > 350
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]

    mdesc0 = np.array([desc[valid] for desc in desc0])
    mdesc1 = np.array([desc[matches0[valid]] for desc in desc1])

    mdesc0 = np.transpose(mdesc0)
    mdesc1 = np.transpose(mdesc1)

    matcher = cv2.BFMatcher_create()
    matches = matcher.match(mdesc0, mdesc1)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]
    pts1 = np.array([mkpts0[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([mkpts1[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    # print('pts1',pts1)
    # print('pts2', pts2)
    img_size = 3.5
    mkpts0_cv2 = [cv2.KeyPoint(x, y, img_size) for x, y in mkpts0]
    mkpts1_cv2 = [cv2.KeyPoint(x, y, img_size) for x, y in mkpts1]

    dst = cv2.drawMatches(img1, mkpts0_cv2, img2, mkpts1_cv2, good_matches,
                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 행렬
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    (h, w) = img1.shape[:2]
    corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(
        np.float32)
    # print('corners1',corners1)
    # print('H', H)
    corners2 = cv2.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    # [[628, -31]], ...
    # 이미지 이쁘게 맞추기
    img_w = int(640)
    img_h = int(480)
    left_up, left_down, right_down, right_up = deepcopy(corners2)
    left_up[0][0] -= img_w
    left_down[0][0] -= img_w
    right_up[0][0] -= img_w
    right_down[0][0] -= img_w

    y_min = min(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
    x_min = min(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])
    y_max = max(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
    x_max = max(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])

    pad_up, pad_down, pad_left, pad_right = 0, 0, 0, 0
    if y_min < 0:
        pad_up = math.ceil(abs(y_min))
    if x_min < 0:
        pad_left = math.ceil(abs(x_min))
    if y_max > img_h:
        pad_down = math.ceil(y_max - img_h)
    if x_max > img_w:
        pad_right = math.ceil(x_max - img_w)

    new_frame = cv2.copyMakeBorder(color, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
    left_up[0][0] += pad_left
    left_down[0][0] += pad_left
    right_up[0][0] += pad_left
    right_down[0][0] += pad_left
    left_up[0][1] += pad_up
    left_down[0][1] += pad_up
    right_up[0][1] += pad_up
    right_down[0][1] += pad_up

    corners3 = np.float32([left_up[0], left_down[0], right_down[0], right_up[0]])
    cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.imwrite('./bbb/h_{:03}_{:03}.png'.format(0, 2), dst)

    ptspts2 = np.float32([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]])
    M = cv2.getPerspectiveTransform(corners3, ptspts2)
    unfold = cv2.warpPerspective(new_frame, M, (img_w, img_h))
    cv2.imwrite('./aaa/h_{:03}_{:03}.png'.format(0, 1), unfold)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        '--resize', type=int, nargs='+', default=[640, 480],
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

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
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

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    img1_t = 'E:\\code\\SuperGluePretrainedNetwork-master\\007.png'
    img2_t = 'E:\\code\\SuperGluePretrainedNetwork-master\\008.png'
    img1 = cv2.imread(img1_t, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(img1_t, cv2.IMREAD_COLOR)
    color = cv2.resize(color, (640, 480))
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.imread(img2_t, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (640, 480))

    perspective(img1, img2, color, get_last_data(img1))




