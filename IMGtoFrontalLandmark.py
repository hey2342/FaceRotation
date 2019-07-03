#coding: utf-8

import face_alignment, cv2, os
import numpy as np
from rotation_lmark import rotation_lmark

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)


DATA_DIR = './014neutral/'
IN_DIR = DATA_DIR + 'Test/'
OUT_DIR = DATA_DIR + 'Test_np/'
dir_names = [x for x in os.listdir(IN_DIR) if not x.startswith('.')]#facial_exp_list
error = open(DATA_DIR +  'error_list_test.txt', 'w')

for d in dir_names:
    img_dir = IN_DIR + d + '/'
    save_dir = OUT_DIR + d + '/'
    if os.path.exists(save_dir) != True:
        os.mkdir(save_dir)

    files = [x for x in os.listdir(img_dir) if not x.startswith('.')]#image list
    for f in files:
        img = cv2.imread(img_dir + f)
        preds = fa.get_landmarks(img)
        if preds != None:
            frontal_lmark = rotation_lmark(preds[0])
            np.save(save_dir + os.path.basename(f)[:-4], frontal_lmark)
        else:
            error.write(d + os.path.basename(f) + '\n')

error.close()
