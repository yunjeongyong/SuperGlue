import os
from PIL import Image
import numpy as np
import cv2
import torch


class LoadFocalFolder(torch.utils.data.Dataset):
    def __init__(self, root, frame_range=None, focal_range=None):
        self.root = root
        # self.frames = os.listdir(root)
        self.frames = [str(i + 1) for i in
                       range(frame_range[0], frame_range[1] + 1)]  # newvideo1: "images/007.png", 65 start frame
        # self.frames = [str(i).zfill(3) for i in range(frame_range[0], frame_range[1]+1)]  #  Nonvideo3: "images/005.png", 33 start
        self.frame_range = frame_range
        # self.type = type  # focal or images
        self.focal_images_path = []
        self.focal_range = focal_range
        self.images_2d_path = []
        self.camera_list = []

        # frame range
        # if self.frame_range is None:
        #     pass
        # else:
        #     self.frames = self.frames[frame_range[0]-int(self.frames[0]): frame_range[1]-int(self.frames[0]) + 1]
# {
#  000: ['///000.png, ///001.png....']
# }
# 000: 7, 000:[...]
        # 2D images setting
        for frame in self.frames:
            type_path = os.path.join(self.root, frame, 'images')
            # image = os.listdir(type_path)[7]
            camera_paths_list = []
            camera_names = os.listdir(type_path)
            camera_names.sort()# 000.png, 001.png...
            for i in range(len(camera_names)):
                camera_paths = os.path.join(type_path, camera_names[i])
                camera_paths_list.append(camera_paths)
            self.camera_list.append(camera_paths_list)
            # print(os.listdir(type_path))
            # image_path = os.path.join(type_path, '005.png')  #  Nonvideo3
            # image_path = os.path.join(type_path, '007.png')  # newvideo1
            # self.images_2d_path.append(image_path)



    def __getitem__(self, idx):
        cameras = self.camera_list[idx]
        images = []
        for camera in cameras:
            img = cv2.imread(camera, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (640, 480))
            images.append(img)

        return images[7], images

    def __len__(self):
        return len(self.camera_list)

        # 2D Images setting

    # def set_images_path(self):
    #     images_path = []
    #     for frame in self.frames:
    #         type_path = os.path.join(self.root, frame, 'images')
    #         image = os.listdir(type_path)[5]
    #         image_path = os.path.join(type_path, image)
    #         images_path.append(image_path)
    #     return images_path


if __name__ == '__main__':
    # dataloader = LoadFocalFolder(root='/ssd2/vot_data/newvideo1/', type='focal', frame_range=(66, 300), focal_range=(0, 100))
    dataloader = LoadFocalFolder(root='D:/dataset/newvideo1', type='focal', frame_range=(66, 300), focal_range=(0, 100))
    print(np.array(dataloader.focal_images_path).shape)
    img = dataloader[0]
    print(img)
