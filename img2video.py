import cv2
import os
import numpy as np
import glob

img_array = []
file_name = []

for filename in glob.glob('D:\\tracker_assignment\\result\\0607_083854\\*.png'):
    file_name.append(filename)
final_output = ['D:\\tracker_assignment\\result\\0607_083854\\' + str(i) for i in sorted([str(num.split('\\')[-1]) for num in file_name])]
print(final_output)
# for filename in glob.glob('D:\\tracker_assignment\\result\\0530_234609\\*.png'):
for filename in final_output:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

print(img_array)
# img_array = ''.join(img_array)
# final_output = [str(i)+".png" for i in sorted([int(num.split('.')[0]) for num in img_array])]
# print(final_output)
# out = cv2.VideoWriter('scenario1.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()


img = img_array[0]
height,width,channel = img.shape
fps = 15

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
writer = cv2.VideoWriter(BASE_DIR + '/' + 'long_output.mp4', fourcc, fps, (width, height))


for file in img_array:

    writer.write(file)

    cv2.imshow("Color", file)

    # ESC키 누르면 중지
    if cv2.waitKey(25) == 27:
        break

writer.release()
cv2.destroyAllWindows()