# coding: utf-8

import face_alignment, cv2, os, sys, copy
from skimage import io
import numpy as np


set_interval = 5

def face_detection(video_dir, dimflag='2d'):
    video_name = os.path.basename(video_dir)[:-4]
    print('=====load video=====')
    video = cv2.VideoCapture(video_dir) 
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    print('=====build output file=====')
    if dimflag == '2d':
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        video_out = cv2.VideoWriter('./detected/' + video_name + '.mp4', fourcc, fps, (width, height)) #2d_output
        face_out = cv2.VideoWriter('./detected/' + video_name + '_face.mp4', fourcc, fps, (320, 320)) #face_output
        lmark_out = cv2.VideoWriter('./detected/' + video_name + '_lmark.mp4', fourcc, fps, (width, height)) #landmark
    elif dimflag == '3d':
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
        video_out = cv2.VideoWriter('./detected_3d/' + video_name + '.mp4', fourcc, fps, (width, height)) #3d_output
        face_out = cv2.VideoWriter('./detected_3d/' + video_name + '_face.mp4', fourcc, fps, (320, 320)) #face_output
        lmark_out = cv2.VideoWriter('./detected_3d/' + video_name + '_lmark.mp4', fourcc, fps, (width, height)) #landmark
    else:
        print('#####dimention key error#####')

    i = -1
    print('=====start face detection======')
    before_lu = np.array([width-1, height-1])
    before_rd = np.array([width, height])
    pred_landmarks = []
    while(video.isOpened()):
        ret, frame = video.read()
        i+=1
        if i % set_interval != 0:
            continue
        if ret == True:
            preds = face_landmark(fa, frame)
            lu, rd = face_box(frame, preds)
            save_face = []
            for j in range(len(lu)):
                #save_image(frame[lu[j][1]:rd[j][1], lu[j][0]:rd[j][0]], str(i)+'_'+str(j))
                if i == 0:
                    if lu[j][0] < before_lu[0]:
                        save_face = cv2.resize(frame[lu[j][1]:rd[j][1], lu[j][0]:rd[j][0]], (320, 320))
                        before_lu = lu[j]
                        before_rd = rd[j]
                else:
                    if np.linalg.norm(before_lu - lu[j]) < 50:
                        save_face = cv2.resize(frame[lu[j][1]:rd[j][1], lu[j][0]:rd[j][0]], (320, 320))
                        before_lu = lu[j]
                        before_rd = rd[j]
                        break
                    else:
                        save_face = cv2.resize(frame[before_lu[1]:before_rd[1], before_lu[0]:before_rd[0]], (320, 320))
            if len(preds) == 0:
                save_face = cv2.resize(frame[before_lu[1]:before_rd[1], before_lu[0]:before_rd[0]], (320, 320))

            #cv2.imwrite('./video/lmark_' + str(i) + '.jpg', lmark_img)
            #if i%500 == 0:
                #if dimflag == '2d':
                    #save_image(image, i)
                #else:
                    #save_image(image, i, '3d')

            new_img = copy.copy(frame)
            lmark_img = plot_alignment(new_img, preds)
         
            video_out.write(image)
            face_out.write(save_face)
            lmark_out.write(lmark_img)
            pred_landmarks.append(preds)
        else:
            break
    video.release()
    video_out.release()
    face_out.release()
    lmark_out.release()
    if dimflag == '2d':
        np.save('./detected/' + video_name, np.array(pred_landmarks))
    elif dimflag == '3d':
        np.save('./detected_3d/' + video_name, np.array(pred_landmarks))
    else:
        print('#####dimention key error#####')


    print('=====finish all=====')
        
def face_landmark(f_alignment, image):
    preds = f_alignment.get_landmarks(image)
    if preds == None:
        preds = []
    return  np.array(preds)
        
def plot_alignment(image, preds):
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            cv2.circle(image, (preds[i][j][0], preds[i][j][1]), 2, (255, 255, 255), thickness=-1)
    return image

def save_image(image, number, dimflag='2d'):
    #print('save ' + str(number))
    if dimflag=='2d':
        cv2.imwrite('./detected/output_' + str(number) + '.jpg', image)
    elif dimflag=='3d':
        cv2.imwrite('./detected_3d/output_' + str(number) + '.jpg', image)


def face_box(image, preds):
    left_up = []
    right_down = []
    for i in range(len(preds)):
        width = [int(min(preds[i, : ,0])), int(max(preds[i, : ,0]))]
        height = [int(min(preds[i, : ,1])), int(max(preds[i, :, 1]))]
        left = 30
        right = 30
        upper = 30
        lower = 20
        if height[0] - upper<0:
            height[0] = 0
        else:
            height[0] -= upper
        if height[1]+lower>image.shape[0]:
            height[1] = image.shape[0]
        else:
            height[1]+=lower
        if width[0] -left<0:
            width[0] = 0
        else:
            width[0] -= left
        if width[1]+right>image.shape[1]:
            width[1] = image.shape[1]
        else:
            width[1]+=right
        left_up.append(np.array([width[0], height[0]]))
        right_down.append(np.array([width[1], height[1]]))
              
    return left_up, right_down

if __name__ == '__main__':
    input_dir = sys.argv[1]
    if len(sys.argv) == 2:
        face_detection(input_dir)
    elif len(sys.argv) == 3:
        face_detection(input_dir, '3d')
    else:
        quit()
