import cv2
import numpy as np
from stitch import Stitch
import sys
import os

def main(input, output):
    img_arr = os.listdir(input)
    for i in range(len(img_arr)):
        img_arr[i] = cv2.imread(os.path.join(input, img_arr[i]))
    panorama = Stitch(img_arr)
    result = panorama.fit_transform()
    final_result = panorama.crop(result)

    cv2.imwrite(os.path.join(output,'panorama.jpg'), final_result)


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Usage: %s <image_dir> <output>' % sys.argv[0])
        print(sys.argv)
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
