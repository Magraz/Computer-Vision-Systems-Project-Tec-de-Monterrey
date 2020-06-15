import cv2
import imutils
import numpy as np

import os
import sys


def get_cnts(img, bb_path):
    with open(bb_path, 'r') as f:
        raw_bbs = f.read().splitlines()

    form_bbs = [[None, None] for _ in range(len(raw_bbs))]
    for i, raw_bb in enumerate(raw_bbs):
        form_bbs[i][0] = raw_bb[0]
        form_bbs[i][1] = [float(b) for b in raw_bb[2:].split(' ')]
    
    h, w = img.shape[:-1]
    
    for i, bb in enumerate(form_bbs):
        pts = bb[1]
        center = np.array(pts[:2])
        center = (center * np.array([w, h])).astype('int')

        bb_w = int(pts[2] * w / 2)
        bb_h = int(pts[3] * h / 2)

        tl = center + np.array([-bb_w, -bb_h])
        tr = center + np.array([bb_w, -bb_h])
        br = center + np.array([bb_w, bb_h])
        bl = center + np.array([-bb_w, bb_h])
        
        bb_cnt = np.array([tl, tr, br, bl], dtype='int')

        form_bbs[i][1] = bb_cnt
    
    return form_bbs


def main(argv):
    BASE_PATH = argv[1]

    try:
        aug_percent = float(argv[2])
    except IndexError:
        aug_percent = 0.8

    try:
        max_rotate = int(argv[3])
    except IndexError:
        max_rotate = 15

    all_files = os.listdir(BASE_PATH)

    file_names = []
    for file in all_files:
        if 'jpg' in file:
            file_names.append(file.split('.')[-2])
    
    np.random.shuffle(file_names)
    n = int((len(file_names) - 1) * aug_percent)
    file_names = file_names[:n]

    for file in file_names:
        IMG_PATH = os.path.join(BASE_PATH, file + '.jpg')
        BB_PATH = os.path.join(BASE_PATH, file + '.txt')
        img = cv2.imread(IMG_PATH)
        h_img, w_img = img.shape[:-1]

        flip = True
        rotate = True
        max_rotate = 15

        # from txt to four cv2 contour points
        cnts = get_cnts(img, BB_PATH)
        bbs_imgs = [np.zeros(img.shape[:-1]) for _ in range(len(cnts))]
        classes = [str(c[0]) for c in cnts]

        # drawing contours in blank canvas
        for im, cnt in zip(bbs_imgs, cnts):
            cv2.drawContours(im, [cnt[1]], -1, 255, 2)

        # augmentation of original img and contour imgs
        rotate_angle = np.random.randint(-max_rotate, max_rotate)
        if flip:
            img = cv2.flip(img, 1)
        if rotate:
            temp_img = imutils.rotate(img, rotate_angle)
            img = imutils.resize(temp_img, width=w_img)

        for i, im in enumerate(bbs_imgs):
            temp_im = im.copy()
            if flip:
                temp_im = cv2.flip(temp_im, 1)
            if rotate:
                temp_im_2 = imutils.rotate(temp_im, rotate_angle)
                temp_im = imutils.resize(temp_im_2, width=w_img)

            bbs_imgs[i] = temp_im

        # extraction of the bounding box from contour images
        new_bbs = []
        for im in bbs_imgs:
            temp_cnt = cv2.findContours(im.astype('uint8'), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            temp_cnt = imutils.grab_contours(temp_cnt)[0]

            box = cv2.boundingRect(temp_cnt)
            box = np.array(box).astype('float')

            box[0] += box[2] / 2
            box[1] += box[3] / 2

            box[0] /= w_img
            box[2] /= w_img
            box[1] /= h_img
            box[3] /= h_img

            new_bbs.append(box)

        IMG_OUT_PATH = os.path.join(BASE_PATH, file + '_aug' + '.jpg')
        cv2.imwrite(IMG_OUT_PATH, img)

        BB_OUT_PATH = os.path.join(BASE_PATH, file + '_aug' + '.txt')
        with open(BB_OUT_PATH, 'w') as f:
            for c, bb in zip(classes, new_bbs):
                f.write(c + ' ' + ' '.join([str(a)[:8] for a in bb]) + '\n')


if __name__ == '__main__':
    main(sys.argv)
