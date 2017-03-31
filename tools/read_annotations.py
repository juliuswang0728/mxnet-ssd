#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import cv2 as cv
from collections import defaultdict

annotations = json.load(open('/home/juliuswang/Desktop/annotations.json'))

out_dir = '../data/plots'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

set_name = 'set01'
base = '/home/juliuswang/Desktop/' + set_name
n_objects = 0
for video_dir in sorted(glob.glob(base + '/*')):
    video_name = os.path.basename(video_dir)
    print video_dir, video_name
    for image_path in sorted(glob.glob(video_dir + '/*.png')):
        image_name = os.path.basename(image_path)
        n_frame = re.search('img([0-9]+)\.png', image_name).groups()[0]
        n_frame = str(int(n_frame))
        if n_frame in annotations[set_name][video_name]['frames']:
            print image_name, annotations[set_name][video_name]['frames'][n_frame]
            data = annotations[set_name][video_name]['frames'][n_frame]
            img = cv.imread(image_path)
            for datum in data:
                x, y, w, h = [int(v) for v in datum['pos']]
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                n_objects += 1
            cv.imshow('annotations', img)
            cv.waitKey(0)

'''img_fns = defaultdict(dict)
image_path = '~/Desktop/set00'
for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    img_fns[set_name] = defaultdict(dict)

for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    img_fns[set_name][video_name] = []

for fn in sorted(glob.glob('data/images/*.png')):
    set_name = re.search('(set[0-9]+)', fn).groups()[0]
    video_name = re.search('(V[0-9]+)', fn).groups()[0]
    n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
    img_fns[set_name][video_name].append((int(n_frame), fn))

n_objects = 0
for set_name in sorted(img_fns.keys()):
    for video_name in sorted(img_fns[set_name].keys()):
        wri = cv.VideoWriter(
            'data/plots/{}_{}.avi'.format(set_name, video_name),
            cv.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        for frame_i, fn in sorted(img_fns[set_name][video_name]):
            img = cv.imread(fn)
            if str(frame_i) in annotations[set_name][video_name]['frames']:
                data = annotations[set_name][
                    video_name]['frames'][str(frame_i)]
                for datum in data:
                    x, y, w, h = [int(v) for v in datum['pos']]
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    n_objects += 1
                wri.write(img)
        wri.release()
        print(set_name, video_name)

print(n_objects)'''