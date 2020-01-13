# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2, sys, os

def rotation_lmark(lmark):
    #右目、左目の重心計算
    reye, leye = eye_pos(lmark)
    #右目、左目、顎でできる平面の法線ベクトルを計算(顔の向き仮)
    cross = np.cross(reye - lmark[8,:], leye - lmark[8,:])
    cross = cross / np.linalg.norm(cross)
    
    #法線ベクトルをxz平面に射影、y軸周りの回転角を計算
    cross_xz = np.array([cross[0], 0, cross[2]])
    cross_xz = cross_xz / np.linalg.norm(cross_xz)
    if cross_xz[0] > 0:
        th_y = rotation_rad(cross_xz, np.array([0, 0, 1]))
    else:
        th_y = -rotation_rad(cross_xz, np.array([0, 0, 1]))
    
    #法線ベクトルをy軸周り回転、x軸周りの回転角を計算
    cross_yz = rotation_y(cross, th_y)
    if cross_yz[1] > 0:
        th_x = -rotation_rad(cross_yz, np.array([0, 0, 1]))
    else:
        th_x = rotation_rad(cross_yz, np.array([0, 0, 1]))
    
    #正面に回転
    r_lmark = []
    for p in lmark:
        r_p = rotation_x(rotation_y(p, th_y), th_x)
        r_lmark.append(r_p)
    r_lmark = np.array(r_lmark)
    
    
    #ランドマークの中心(鼻の頭を原点に移動)
    r_lmark = r_lmark - r_lmark[33, :]

    #右目と左目の位置からz軸周りの回転角を計算
    reye, leye = eye_pos(r_lmark)
    rot_z = np.array([reye[0]-leye[0], reye[1]-leye[1], 0])
    rot_z = rot_z / np.linalg.norm(rot_z)
    rot_z = np.array([rot_z[1], -rot_z[0], 0])
    if rot_z[0] > 0:
        th_z = -rotation_rad(rot_z, np.array([0, 1, 0]))
    else:
        th_z = rotation_rad(rot_z, np.array([0, 1, 0]))
    #z軸周りで回転
    out_lmark = []
    for p in r_lmark:
        r_p = rotation_z(p, th_z)
        out_lmark.append(r_p)
            
    return normalize_lmark(np.array(out_lmark))

def eye_pos(lmark):
    reye = np.average(lmark[36:42], axis=0)
    leye = np.average(lmark[42:48], axis=0)

    return reye, leye

def rotation_rad(cross, axis):
    c = np.inner(cross, axis)
    rad = np.arccos(np.clip(c, -1.0, 1.0))
    
    return rad

def rotation_x(vec, th):
    x_rotM = np.array([[1, 0, 0],\
                       [0, np.cos(th), np.sin(th)],\
                       [0, -np.sin(th), np.cos(th)]])
    
    return np.dot(x_rotM, vec)
    
def rotation_y(vec, th):
    y_rotM = np.array([[np.cos(th), 0, -np.sin(th)],\
                       [0, 1, 0],\
                       [np.sin(th), 0, np.cos(th)]])
    
    return np.dot(y_rotM, vec)

def rotation_z(vec, th):
    z_rotM = np.array([[np.cos(th), np.sin(th), 0],\
                       [-np.sin(th), np.cos(th), 0],\
                       [0, 0, 1]])
    
    return np.dot(z_rotM, vec)

def normalize_lmark(lmark):
    minW = min(lmark[:, 0])
    minH = min(lmark[:, 1])
    maxD = max(lmark[:, 2])
    BB_origin = [minW,  minH, maxD]
    norm_lmark = lmark-BB_origin
    maxH = max(norm_lmark[:, 1])
    stretch = 100/maxH
    return norm_lmark * stretch
    

if __name__ == '__main__':
    in_dir = sys.argv[1]
    lmarks = np.load(in_dir)
    if 'fixed' in in_dir:
        name = os.path.basename(in_dir).replace('_fixed.npy', '')
    else:
        name = os.path.basename(in_dir).replace('.npy', '')
    out_dir = 'rotated/' + name + '/npy/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out = []
    all_frame = len(str(int(len(lmarks)*5)))
    for i, lmark in enumerate(lmarks):
        if len(lmark) > 0:
            r_lmark = rotation_lmark(lmark[0])
            np.save(out_dir + str(i*5).zfill(all_frame), r_lmark)
            out.append(r_lmark)
        else:
            out.append([])
    #np.save('all_face', np.array(out))       
    #fig = plt.figure()
    #rX = r_lmark[:,0]
    #rY = r_lmark[:,1]
    #rZ = r_lmark[:,2]
    #plt.scatter(rX, -rY)
    #plt.show()

