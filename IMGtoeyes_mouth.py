#coding: utf-8

import face_alignment, cv2, os, sys, copy
import numpy as np
from rotation_lmark import rotation_lmark
from FaceNormalization import img_normalization

DATA_DIR = sys.argv[1]
IN_DIR = DATA_DIR  +  'data/'
FACE_DIR = DATA_DIR + 'frontal/'
REYE_DIR = DATA_DIR + 'reye/'
LEYE_DIR = DATA_DIR  + 'leye/'
MOUTH_DIR = DATA_DIR + 'mouth/'
LMARK_DIR = DATA_DIR + 'lmark/'
FLMARK_DIR = DATA_DIR + 'frontal_lmark/'


if __name__ == '__main__':
    dir_names = [x for x in os.listdir(IN_DIR) if not x.startswith('.')]#facial_exp_list
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    for d in dir_names:
        img_dir = IN_DIR + d + '/'
        frontal_dir = FACE_DIR + d + '/'
        reye_dir = REYE_DIR + d + '/'
        leye_dir = LEYE_DIR + d + '/'
        mouth_dir = MOUTH_DIR + d + '/'
        lmark_dir = LMARK_DIR + d + '/'
        flmark_dir = FLMARK_DIR + d + '/'
        if not os.path.exists(frontal_dir):
            os.mkdir(frontal_dir)
        if not os.path.exists(reye_dir):
            os.mkdir(reye_dir)
        if not os.path.exists(leye_dir):
            os.mkdir(leye_dir)
        if not os.path.exists(mouth_dir):
            os.mkdir(mouth_dir)
        if not os.path.exists(lmark_dir):
            os.mkdir(lmark_dir)
        if not os.path.exists(flmark_dir):
            os.mkdir(flmark_dir)

        files = [x for x in os.listdir(img_dir) if not x.startswith('.')]#image list
        for f in files:
            if os.path.exists(frontal_dir + os.path.basename(f)[:-4] + '.png'):
                continue
            print(img_dir + f)
            img = cv2.imread(img_dir + f)
            preds = fa.get_landmarks(img_dir + f)
            #preds = np.load('pred_alignment/'+d+'/'+ f[:-3] + 'npy')

            if len(preds) > 0:
                in_preds = copy.copy(preds[0][:,:2])
                frontal, reye, leye, mouth, f_lmark = img_normalization(img, in_preds)

                cv2.imwrite(frontal_dir + os.path.basename(f)[:-4] + '.png', frontal)
                cv2.imwrite(reye_dir + os.path.basename(f)[:-4] + '.png', reye)
                cv2.imwrite(leye_dir + os.path.basename(f)[:-4] + '.png', leye)
                cv2.imwrite(mouth_dir + os.path.basename(f)[:-4] + '.png', mouth)

                for i in range(len(preds[0])):
                    cv2.circle(img, (preds[0][i][0], preds[0][i][1]), 2, (0, 0, 255), thickness=-1)
                cv2.imwrite(lmark_dir + os.path.basename(f)[:-4] + '.png', img)
                cv2.imwrite(flmark_dir + os.path.basename(f)[:-4] + '.png', f_lmark)
                
            else:
                print('No face : ' + d + os.path.basename(f))
