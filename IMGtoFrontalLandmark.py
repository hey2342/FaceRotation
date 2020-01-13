#coding: utf-8

import face_alignment, cv2, os, sys
import numpy as np
from rotation_lmark import rotation_lmark

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

IN_DIR = sys.argv[1]    #F0001/data/
OUT_DIR = sys.argv[2]   #F0001/np/
dir_names = [x for x in os.listdir(IN_DIR) if not x.startswith('.')]#facial_exp_list
#error = open(DATA_DIR +  'error_list_test.txt', 'w')

for d in dir_names:
    img_dir = IN_DIR + d + '/'
    save_dir = OUT_DIR + d + '/'
    if os.path.exists(save_dir) != True:
        os.mkdir(save_dir)
        #os.mkdir('pred_alignment/' + d)

    files = [x for x in os.listdir(img_dir) if not x.startswith('.')]#image list
    for f in files:
        if os.path.exists(save_dir + os.path.basename(f)[:-4]+'.npy'):
            continue
        print(img_dir + f)
        #img = cv2.imread(img_dir + f)
        #preds = fa.get_landmarks(img)
        preds = fa.get_landmarks(img_dir + f)
        if preds != None:
            frontal_lmark = rotation_lmark(preds[0])
            #np.save('pred_alignment/' +d+'/'+ os.path.basename(f)[:-4], preds)
            np.save(save_dir + os.path.basename(f)[:-4], frontal_lmark)
        else:
            print('No Faces : ' + d + os.path.basename(f))
            #error.write(d + os.path.basename(f) + '\n')

#error.close()
