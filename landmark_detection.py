import sys
import numpy as np


lmark_path = sys.argv[1]
set_start = int(sys.argv[2])
MIN_AREA = 200
MAX_AREA = 1160

org_lmarks =  np.load(lmark_path)

start_pos = 60*set_start
print('start from : ', start_pos)

def lmarks_to_rect_area(lmarks):
    face_pos = []
    face_area = []
    for lmark in lmarks:
        face_pos_lu = [min(lmark[:,0]), min(lmark[:,1])] #landmark left_up
        face_pos_rd = [max(lmark[:,0]), max(lmark[:,1])] #landmark right_down
        width = face_pos_rd[0] - face_pos_lu[0]
        height = face_pos_rd[1] - face_pos_lu[1]
        area = width * height

        face_pos.append(face_pos_lu)
        face_area.append(area)
    return face_pos, face_area

out_lmarks = []
for i, lmarks in enumerate(org_lmarks):
    if len(lmarks) == 0 or i < start_pos:
        out_lmarks.append([])
    else:
        if all(x==[] for x in out_lmarks):
            target_id = np.argmin(np.array(lmarks[:,0,0])) #decide target
            before_face = lmarks[target_id] #target's landmark
            face_posls, face_areals = lmarks_to_rect_area([before_face])
            before_pos = np.array(face_posls[0])
            before_area = face_areals[0]
            out_lmarks.append(before_face)
        else:
            face_posls, face_areals = lmarks_to_rect_area(lmarks)
            comp_pos = [np.linalg.norm(x - before_pos) for x in face_posls]
            comp_area = [x / before_area >= 0.9 and x / before_area <= 1.1 for x in face_areals] #0.9 <= current_area/before_area  <= 1.1
            #prob_target = []
            #for j, c_a in enumerate(comp_area):
                #if c_a:
                    #prob_target.append(comp_pos[j])
                #else:
                    #prob_target.append(1000)
            prob_target = np.array(comp_pos)
            while True:
                if len(prob_target) == 0:
                    out_lmarks.append([])
                    break
                else:
                    target_id = np.argmin(prob_target)
                    if face_posls[target_id][0]>MAX_AREA and face_posls[target_id][0]<MIN_AREA:
                        prob_target = np.delete(prob_target, target_id)
                    else:
                        before_pos = np.array(face_posls[target_id])
                        before_area = face_areals[target_id]
                        out_lmarks.append(lmarks[target_id])
                        break

np.save(sys.argv[1][:-4] + '_fixed', np.array(out_lmarks))
