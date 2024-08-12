import cv2
import numpy as np
from copy import deepcopy
import math
from dfasdfsad_dataloader import LoaddFocalFolder

input = '/media/mnt/dataset'
new = 'H_result_5.npy'
dataset = LoaddFocalFolder(input, new, frame_range=[0, 601], focal_range=None)
# img = '/media/mnt/dataset/1/images/002.png'
for idx in range(dataset.__len__()):
    frames, H_list = dataset.__getitem__(idx)
    for frame_idx, (frame, H) in enumerate(zip(frames, H_list)):
        print('frame_idx', frame_idx)
        print('len_H',len(H))
        print('H_list_len',len(H_list))
        print('H',H)
        print('frame',frame)
        (h, w) = frame.shape[:2]
        print('h',h)
        print('w',w)
        corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
        # print('corners1',corners1)
        # print('H', H)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners2 = corners2 + np.float32([w, 0])

        # [[628, -31]], ...
        left_up, left_down, right_down, right_up = deepcopy(corners2)
        # left_up[0][0] -= 3840
        # left_down[0][0] -= 3840
        # right_up[0][0] -= 3840
        # right_down[0][0] -= 3840
        #
        # y_min = min(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
        # x_min = min(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])
        # y_max = max(left_up[0][1], left_down[0][1], right_up[0][1], right_down[0][1])
        # x_max = max(left_up[0][0], left_down[0][0], right_up[0][0], right_down[0][0])
        #
        pad_up, pad_down, pad_left, pad_right = 0, 0, 0, 0
        # if y_min < 0:
        #     pad_up = math.ceil(abs(y_min))
        # if x_min < 0:
        #     pad_left = math.ceil(abs(x_min))
        # if y_max > 2160:
        #     pad_down = math.ceil(y_max - 2160)
        # if x_max > 3840:
        #     pad_right = math.ceil(x_max - 3840)
        #
        #
        new_frame = cv2.copyMakeBorder(frame, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # left_up[0][0] += pad_left
        # left_down[0][0] += pad_left
        # right_up[0][0] += pad_left
        # right_down[0][0] += pad_left
        # left_up[0][1] += pad_up
        # left_down[0][1] += pad_up
        # right_up[0][1] += pad_up
        # right_down[0][1] += pad_up

        corners3 = np.float32([left_up[0], left_down[0], right_down[0], right_up[0]])
        corners4 = np.int32(corners3)

        a = np.float32([left_up[0], left_down[0], right_down[0], right_up[0]])
        b = deepcopy(new_frame)
        cv2.polylines(b, [np.int32(corners3)], True, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite('./sample.png', b)
        # cv2.imwrite('./homographhhhhy/h_{:03}_{:03}.png'.format(idx, frame_idx), b)

        # cv2.polylines(new_frame, [np.int32(corners3)], True, (0, 255, 0), 2, cv2.LINE_AA)
        # # cv2.imwrite('./newframe.png', new_frame)
        # cv2.imwrite('./newframe/h_{:03}_{:03}.png'.format(idx, frame_idx), new_frame)

        # ptspts2 = np.float32([[0, 0], [0, 2160], [3840, 2160], [3840, 0]])
        # M = cv2.getPerspectiveTransform(corners3, ptspts2)
        # unfold = cv2.warpPerspective(new_frame, M, (3840, 2160))
        # cv2.imwrite('./before_warp/h_{:03}_{:03}.png'.format(idx, frame_idx), unfold)
