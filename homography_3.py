import numpy as np
import cv2
from models.utils_dataloader import LoadFocalFolder
from models.utils_dataloader_02 import LoadFocalFolderColorAndGrey
import matplotlib.cm as cm
import torch
import sys
from copy import deepcopy
import math

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)


keys = ['keypoints', 'scores', 'descriptors']


def get_last_data(frame):
    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    return last_data


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
folder = 'D:/newvideo1_001_300'
dataset = LoadFocalFolderColorAndGrey(folder, frame_range=[0, 601], focal_range=None)
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

for idx in range(dataset.__len__()):
    last_frame, frames, colors = dataset.__getitem__(idx)
    print('last_frame', last_frame.shape)
    for frame_idx, (frame, color) in enumerate(zip(frames, colors)):
        stem0, stem1 = 7, frame_idx
        last_data = get_last_data(last_frame)

        frame_tensor = frame2tensor(frame, device)
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
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        print(H)

        img_size = 3.5
        mkpts0_cv2 = [cv2.KeyPoint(x, y, img_size) for x, y in mkpts0]
        mkpts1_cv2 = [cv2.KeyPoint(x, y, img_size) for x, y in mkpts1]

        dst = cv2.drawMatches(last_frame, mkpts0_cv2, frame, mkpts1_cv2, good_matches,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        (h, w) = last_frame.shape[:2]
        corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
        # print('corners1',corners1)
        # print('H', H)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners2 = corners2 + np.float32([w, 0])

        # [[628, -31]], ...
        left_up, left_down, right_down, right_up = deepcopy(corners2)
        left_up[0][0] -= 640
        left_down[0][0] -= 640
        right_up[0][0] -= 640
        right_down[0][0] -= 640

        y_min = min(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
        x_min = min(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])
        y_max = max(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
        x_max = max(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])

        pad_up, pad_down, pad_left, pad_right = 0, 0, 0, 0
        if y_min < 0:
            pad_up = math.ceil(abs(y_min))
        if x_min < 0:
            pad_left = math.ceil(abs(x_min))
        if y_max > 480:
            pad_down = math.ceil(y_max - 480)
        if x_max > 640:
            pad_right = math.ceil(x_max - 640)

        # print(color)
        # print(pad_up)
        # print(pad_down)
        # print(pad_left)
        # print(pad_right)

        new_frame = cv2.copyMakeBorder(color, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        left_up[0][0] += pad_left
        left_down[0][0] += pad_left
        right_up[0][0] += pad_left
        right_down[0][0] += pad_left
        left_up[0][1] += pad_up
        left_down[0][1] += pad_up
        right_up[0][1] += pad_up
        right_down[0][1] += pad_up

        corners3 = np.float32([left_up[0], left_down[0], right_down[0], right_up[0]])
        corners4 = np.int32(corners3)

        # a = np.float32([left_up[0], left_down[0], right_down[0], right_up[0]])
        # b = deepcopy(new_frame)
        # b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        # cv2.polylines(b, [np.int32(corners3)], True, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite('./sample.png', b)

        # cv2.polylines(new_frame, [np.int32(corners3)], True, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite('./newframe.png', new_frame)

        ptspts2 = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
        M = cv2.getPerspectiveTransform(corners3, ptspts2)
        unfold = cv2.warpPerspective(new_frame, M, (640, 480))
        cv2.imwrite('./unfold_images_color_1@/h_{:03}_{:03}.png'.format(idx, frame_idx), unfold)

        # concat_unfold = cv2.hconcat([last_frame, unfold])
        # cv2.imwrite('./concat_unfold_images_@/h_{:03}_{:03}.png'.format(idx, frame_idx), concat_unfold)
        #
        # cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        # # cv2.imshow('dst', dst)
        # cv2.imwrite('./homography_images_@/h_{:03}_{:03}.png'.format(idx, frame_idx), dst)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        #
        # sys.exit(0)
        # if frame_idx == 2:
        #     sys.exit(0)

# cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# src1 = cv2.imread('007.png', cv2.IMREAD_GRAYSCALE)
# src2 = cv2.imread('015.png', cv2.IMREAD_GRAYSCALE)
# sr1 = cv2.resize(src1, (128, 56))
# src2 = cv2.resize(src2, (128, 56))
# feature = cv2.KAZE_create()
# kp1, desc1 = feature.detectAndCompute(src1, None)
# kp2, desc2 = feature.detectAndCompute(src2, None)
# matcher = cv2.BFMatcher_create()
# matches = matcher.match(desc1, desc2)
# matches = sorted(matches, key=lambda x: x.distance)
# good_matches = matches[:80]
# pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
# pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
# H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
# dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None,
# flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# (h, w) = src1.shape[:2]
# corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
# corners2 = cv2.perspectiveTransform(corners1, H)
# corners2 = corners2 + np.float32([w, 0])
#
# cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()
