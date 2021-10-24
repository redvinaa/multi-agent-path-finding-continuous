#! /usr/bin/env python3.6

import cv2
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map_name', type=str, help='Which map to open')
    parser.add_argument('-s', '--save', action='store_true', help='Save resized pic as jpg')
    args = parser.parse_args()

    maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps')
    map_path = os.path.join(maps_dir, args.map_name+'.jpg')

    if not os.path.isfile(map_path):
        print('Not a valid map name')
        quit()

    pic = cv2.imread(map_path)
    pic = cv2.resize(pic, (700, 700), interpolation=cv2.INTER_AREA)

    if args.save:
        new_map_path = os.path.join(maps_dir, args.map_name+'_resized.jpg')
        cv2.imwrite(new_map_path, pic)

    cv2.imshow('Map', pic)
    cv2.waitKey()
