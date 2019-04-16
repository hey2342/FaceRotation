# coding: utf-8

import face_alignment, cv2, os, sys
from skimage import io
import numpy as np




def face_detection(video_dir, dimflag='2d'):
    video_name = os.path.basename(video_dir)[:-3]   #file name
    print('=====load video=====')
    video = cv2.VideoCapture('./video/' + video_dir)        #video obj 
    width = int(video.get(3))             #video width
    height = int(video.get(4))            #video height
    fps = video.get(5)                    #video fps
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    print('=====build output file=====')
    if dimflag == '2d':
        video_out = cv2.VideoWriter('./detected/' + video_name + 'mp4', fourcc, fps, (width, height)) #2d_output
    elif dimflag == '3d':
        video_out = cv2.VideoWriter('./detected_3d/' + video_name + 'mp4', fourcc, fps, (width, height)) #3d_output
    else:
        print('#####3d key error#####')

    i = -1
    print('=====start face detection======')
    while(video.isOpened()):
        ret, frame = video.read()
        i+=1
        if i % 5 != 0:
            continue
        if ret == True:
            preds = face_landmark(frame)
            w, h = face_box(frame, preds)
            for j in range(len(w)):
                save_image(frame[h[j][0]:h[j][1], w[j][0]:w[j][1]], str(i)+'_'+str(j))
            image = plot_alignment(frame, preds)
            #if i%500 == 0:
                #if dimflag == '2d':
                    #save_image(image, i)
                #else:
                    #save_image(image, i, '3d')
            video_out.write(image)
        else:
            break
    video.release()
    video_out.release()
    print('=====finish all=====')
        
def face_landmark(image):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    preds = fa.get_landmarks(image)
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
    widthls = []
    heightls = []
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
        widthls.append(width)
        heightls.append(height)    
              

    return widthls, heightls

if __name__ == '__main__':
    input_dir = sys.argv[1]
    if len(sys.argv) == 2:
        face_detection(input_dir)
    elif len(sys.argv) == 3:
        face_detection(input_dir, '3d')
    else:
        quit()
