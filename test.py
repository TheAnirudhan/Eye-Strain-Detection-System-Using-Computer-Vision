from main import *
import cv2
import numpy as np
import matplotlib.pyplot as plt


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)
ratio_l =[]
l_gaze_list=[]
r_gaze_list=[]
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if not ret: # frame not read successfully
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #Mediapipe precisa do formato de cores RGB mas o OpenCV usa o BGR
        frame_height, frame_width = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        for facial_landmarks in results.multi_face_landmarks:
            mesh_coords=np.array([np.multiply([p.x, p.y], [frame_width, frame_height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            iris = irisSize(mesh_coords,LEFT_IRIS,RIGHT_IRIS)
            Edd = computeEyeDeviceDistance(iris)
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_coords[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_coords[RIGHT_IRIS])
            
            for p in GLABELLAR:
                cv.circle(frame, mesh_coords[p] ,2,(255,0,0),-1)
            glab = glabLength(mesh_coords,GLABELLAR)
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            l_eye =mesh_coords[LEFT_EYE[0]]
            r_eye = mesh_coords[RIGHT_EYE[0]]
            l_eye_gaze = [abs(l_cx -l_eye[0]), abs(l_cy - l_eye[1])]
            r_eye_gaze = [abs(r_cx -r_eye[0]), abs(r_cy - r_eye[1])]
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in GLABELLAR[0:2] ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in GLABELLAR[2:4] ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in GLABELLAR[4:6] ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            l_gaze_list.append(l_eye_gaze)
            r_gaze_list.append(r_eye_gaze)
            #cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            #cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            utils.colorBackgroundText(frame,  f'Glabellar Length  : {round(glab,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
            
            '''
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE) # EAR
            ratio_l.append(ratio)    
            if ratio < EAR_THR:
                CEF_COUNTER +=1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)    
            
            pt1 = facial_landmarks.landmark[474]
            x1 = int(pt1.x * img_w)
            y1 = int(pt1.y * img_h)           
            cv2.circle(frame, (x1,y1),2,(255,0,0),-1)
            pt2 = facial_landmarks.landmark[47]
            x2 = int(pt2.x * img_w)
            y2 = int(pt2.y * img_h)    
            
            
            cv2.circle(frame, (x2,y2),2,(255,0,0),-1)
            print(cv2.norm((x1,y1),(x2,y2)))
            
            for i in range(0,468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * img_w)
                y = int(pt1.y * img_h)
                cv2.circle(frame, (x,y),2,(255,0,0),-1)
            '''
        Emd = computeEyeMovementDuration(l_gaze_list, r_gaze_list)
        cv2.imshow("img", frame)
        key = cv2.waitKey(1)
        if key ==ord("q"):
            break
'''
plt.plot()
plt.xlabel('Video Frames')
plt.ylabel('Eye Movement Delta')
plt.show()'''