#coding: utf-8

import face_alignment, cv2, os, sys
import numpy as np
from rotation_lmark import rotation_lmark

MODE = sys.argv[1] #Train or Test

DATA_DIR = './MUGdataset/'
IN_DIR = DATA_DIR + MODE +  '/'
REYE_DIR = DATA_DIR + MODE + '_reye/'
LEYE_DIR = DATA_DIR + MODE + '_leye/'
MOUTH_DIR = DATA_DIR + MODE + '_mouth/'

EYE_SIZE = [128, 64]
MOUTH_SIZE = [256, 128]

def lmark_bbox(lmark, pos = 'eye'):
    center = [sum(lmark[:,0])/len(lmark), sum(lmark[:,1])/len(lmark)]
    half_eye = [EYE_SIZE[0]/2, EYE_SIZE[1]/2]
    half_mouth = [MOUTH_SIZE[0]/2, MOUTH_SIZE[1]/2]
    if pos == 'eye':
        up_left = [int(center[1]-half_eye[1]), int(center[0]-half_eye[0])]
        down_right = [int(center[1]+half_eye[1]), int(center[0]+half_eye[0])]
    if pos == 'mouth':
        up_left = [int(center[1]-half_mouth[1]), int(center[0]-half_mouth[0])]
        down_right = [int(center[1]+half_mouth[1]), int(center[0]+half_mouth[0])]
    return up_left, down_right


if __name__ == '__main__':
    dir_names = [x for x in os.listdir(IN_DIR) if not x.startswith('.')]#facial_exp_list
    error = open(DATA_DIR +  'error_list_test.txt', 'w')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    for d in dir_names:
        img_dir = IN_DIR + d + '/'
        reye_dir = REYE_DIR + d + '/'
        leye_dir = LEYE_DIR + d + '/'
        mouth_dir = MOUTH_DIR + d + '/'
        if os.path.exists(reye_dir) != True:
            os.mkdir(reye_dir)
        if os.path.exists(leye_dir) != True:
            os.mkdir(leye_dir)
        if os.path.exists(mouth_dir) != True:
            os.mkdir(mouth_dir)

        files = [x for x in os.listdir(img_dir) if not x.startswith('.')]#image list
        for f in files:
            img = cv2.imread(img_dir + f)
            preds = fa.get_landmarks(img)
            if preds != None:
                #############frontal_face = 
                reye_ul, reye_dr = lmark_bbox(preds[0][36:42])
                reye = img[reye_ul[0]:reye_dr[0], reye_ul[1]:reye_dr[1]]
                cv2.imwrite(reye_dir + os.path.basename(f)[:-4] + '.jpg', reye)
                
                leye_ul, leye_dr = lmark_bbox(preds[0][42:48])
                leye = img[leye_ul[0]:leye_dr[0], leye_ul[1]:leye_dr[1]]
                cv2.imwrite(leye_dir + os.path.basename(f)[:-4] + '.jpg', leye)
                
                mouth_ul, mouth_dr = lmark_bbox(preds[0][48:68], pos = 'mouth')
                mouth = img[mouth_ul[0]:mouth_dr[0], mouth_ul[1]:mouth_dr[1]]
                cv2.imwrite(mouth_dir + os.path.basename(f)[:-4] + '.jpg', mouth)
            else:
                error.write(d + os.path.basename(f) + '\n')

    error.close()


