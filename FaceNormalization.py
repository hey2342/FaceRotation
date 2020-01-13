# coding: utf-8

import face_alignment, cv2, os, sys
import numpy as np

import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import check_resources as check
import matplotlib.pyplot as plt
import copy

this_path = os.path.dirname(os.path.abspath(__file__)) 
EYE_SIZE = [32, 16]
MOUTH_SIZE = [64, 32]


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


def find_nearest(mat, value):
    face_lr = [100, 220]
    face_ud = [60, 230]
    L = np.zeros((face_ud[1]-face_ud[0], face_lr[1]-face_lr[0]))
    for i in range(face_ud[0], face_ud[1]):
        for j in range(face_lr[0], face_lr[1]):
            L[i-face_ud[0]][j-face_lr[0]] = np.linalg.norm(mat[i][j]-value)
    nearest = np.unravel_index(np.argmin(L), L.shape)
    return nearest[0] + face_ud[0], nearest[1] + face_lr[0]


def img_normalization(img, lmark):
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])

 
    height, width, _ = img.shape

    #face_lu = np.array([min(lmark[:,0]), min(lmark[:,1])])
    #face_rd = np.array([max(lmark[:,0]), max(lmark[:,1])])
    #face_lu = face_lu - 20
    #face_rd = face_rd + 20

    #if face_lu[0]<0:
    #    face_lu[0] = 0
    #if face_lu[1]<0:
    #    face_lu[1] = 0
    #if face_rd[0] > width:
    #    face_rd[0] = width
    #if face_rd[1] > height:
    #    face_rd[1] = height
    #face_lu = face_lu.astype('uint64')
    #face_rd = face_rd.astype('uint64')
 
    #new_img = img[face_lu[1]:face_rd[1], face_lu[0]:face_rd[0]]
    #new_img = cv2.resize(img, (320, 320))

    #lmark = lmark - [face_lu[0], face_lu[1]]
    #w_res = 320/(face_rd[0]-face_lu[0])
    #h_res = 320/(face_rd[1]-face_lu[1])
    #lmark = lmark*[w_res, h_res]
    #new_lm = copy.copy(lmark)
            
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmark)
    frontal_raw, frontal_sym, proj_map  = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    
    rot_lmark = np.zeros((lmark.shape[0], 2))
    for i in range(lmark.shape[0]):
        rot_lmark[i][1], rot_lmark[i][0] = find_nearest(proj_map, lmark[i])
 
    reye_ul, reye_dr = lmark_bbox(rot_lmark[36:42])
    leye_ul, leye_dr = lmark_bbox(rot_lmark[42:48])
    mouth_ul, mouth_dr = lmark_bbox(rot_lmark[48:68], pos = 'mouth')

    int_frontal = frontal_sym.astype('uint8')
    reye = int_frontal[reye_ul[0]:reye_dr[0], reye_ul[1]:reye_dr[1]]
    leye = int_frontal[leye_ul[0]:leye_dr[0], leye_ul[1]:leye_dr[1]]
    mouth = int_frontal[mouth_ul[0]:mouth_dr[0], mouth_ul[1]:mouth_dr[1]]

    frontal_lmark = np.copy(int_frontal)
    for i in range(len(rot_lmark)):
        cv2.circle(frontal_lmark, (int(rot_lmark[i][0]), int(rot_lmark[i][1])), 2, (0, 0, 255), thickness=-1)
    
    return int_frontal, reye, leye, mouth, frontal_lmark


if __name__ == '__main__':
    print('=== loading video ===')
    video_dir = sys.argv[1]
    video_name = os.path.basename(video_dir)[:-4]
    video = cv2.VideoCapture(video_dir) 
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    frame_len = len(str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))*5))

    #save_dir
    if not os.path.exists('./rotated/' + video_name):
        os.mkdir('./rotated/' + video_name)
    if not os.path.exists('./rotated/' + video_name + '/leye'):
        os.mkdir('./rotated/' + video_name + '/leye')
    if not os.path.exists('./rotated/' + video_name + '/reye'):
        os.mkdir('./rotated/' + video_name + '/reye')
    if not os.path.exists('./rotated/' + video_name + '/mouth'):
        os.mkdir('./rotated/' + video_name + '/mouth')

    video_out = cv2.VideoWriter('./rotated/' + video_name + '.mp4', fourcc, fps, (320, 320)) #3d_output
    leye_out = cv2.VideoWriter('./rotated/' + video_name + '_leye.mp4', fourcc, fps, (EYE_SIZE[0], EYE_SIZE[1])) #leye
    reye_out = cv2.VideoWriter('./rotated/' + video_name + '_reye.mp4', fourcc, fps, (EYE_SIZE[0], EYE_SIZE[1])) #reye
    mouth_out = cv2.VideoWriter('./rotated/' + video_name + '_mouth.mp4', fourcc, fps, (MOUTH_SIZE[0], MOUTH_SIZE[1])) #mouth

    print('=== loading landmarks ===')
    lmarks = np.load(sys.argv[2])
    #lmarks = np.load('./detected_3d/' + video_name + '_fixed.npy')

    print('===loading model === ')
    check.check_dlib_landmark_weights()
    black_img = np.zeros((320, 320,3), dtype='uint8')

    i=-1
    while(video.isOpened() and i+1 < len(lmarks)):
        ret, frame = video.read()
        i+=1
        print('reading : frame ' + str(i))

        if ret == True and len(lmarks[i]) > 0:
            lmark = copy.copy(lmarks[i][0][:,:2])
            frontal, reye, leye, mouth, _ = img_normalization(frame, lmark)

            video_out.write(frontal)
            reye_out.write(reye)
            leye_out.write(leye)
            mouth_out.write(mouth)
        
            cv2.imwrite('./rotated/' + video_name + '/frontal_sym_' + str(i*5).zfill(frame_len) + '.png', frontal)
            cv2.imwrite('./rotated/' + video_name + '/reye/' + str(i*5).zfill(frame_len) + '.png', reye)
            cv2.imwrite('./rotated/' + video_name + '/leye/' + str(i*5).zfill(frame_len) + '.png', leye)
            cv2.imwrite('./rotated/' + video_name + '/mouth/' + str(i*5).zfill(frame_len) + '.png', mouth)
        elif ret == True and len(lmarks[i]) <= 0:
            video_out.write(black_img)
            reye_out.write(black_img[:EYE_SIZE[1],:EYE_SIZE[0],:])
            leye_out.write(black_img[:EYE_SIZE[1],:EYE_SIZE[0],:])
            mouth_out.write(black_img[:MOUTH_SIZE[1],:MOUTH_SIZE[0],:])
        else:
            pass


    video_out.release()
    reye_out.release()
    leye_out.release()
    mouth_out.release()
                    
