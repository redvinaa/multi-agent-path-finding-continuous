#! /usr/bin/env python3.6

import numpy as np
from rospkg import RosPack
import os
import cv2


name = 'test1_4x4.jpg'
img = np.array([[255, 255, 255, 255],
                [255, 255, 255, 0],
                [0,   255, 255, 255],
                [255, 255, 255, 255]], dtype=np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

img_big = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)

cv2.imshow(f"New map: {name}", img_big)
cv2.waitKey()

pkg_path = RosPack().get_path('mapf_environment')
img_path = os.path.join(pkg_path, 'maps', name)

if (input(f'Save to {img_path}? (y/n)') == 'y'):
    cv2.imwrite(img_path, img)
