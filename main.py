import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import saccademodel

# global variables
TEST_DURATION = 3
RECCURENCE_INTERVAL = 20 

EAR_THR  = 0.22
OPEN_EYE = 0.5
GLAB_THR = 12
IRIS_SIZE_PX = 10
FOCAL = 35

frame_counter = 0
CEF_COUNTER =0
TOTAL_BLINKS =0


# constants
CLOSED_EYES_FRAME = 3
FONTS =cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
GLABELLAR = [8,6,55,189,285,417]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(0)
# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):

    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    #reRatio = rhDistance/rvDistance
    reRatio = rvDistance/rhDistance
    #leRatio = lhDistance/lvDistance
    leRatio = lvDistance/lhDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def glabLength(landmarks, glab):
    l1 = cv.norm(landmarks[glab[0]],landmarks[glab[1]])
    l2 = cv.norm(landmarks[glab[2]],landmarks[glab[3]])
    l3 = cv.norm(landmarks[glab[4]],landmarks[glab[5]])
    return ((l1+l2+l3)/3)

def irisSize(landmarks, left_iris, right_iris):
    l_size = euclaideanDistance(landmarks[left_iris[0]], landmarks[left_iris[2]])
    r_size = euclaideanDistance(landmarks[right_iris[0]],landmarks[right_iris[2]])
    return ((l_size+ r_size) / 2)
 
def computeBlinkRate(duration):
    global TOTAL_BLINKS
    return (TOTAL_BLINKS*60/duration)

def computeSquintEyeDuration(ear):
    global OPEN_EYE
    ear=np.array(ear)
    squint_list = ear[ear < 0.6*OPEN_EYE ]
    return (len(squint_list)/len(ear))

def computeEyeDeviceDistance(iris_list):
    iris_list = np.array(iris_list)
    return (np.mean(FOCAL*(12/iris_list)+10))

def computeEyeMovementDuration(l_gaze, r_gaze):
    l_gaze_res = saccademodel.fit(l_gaze)
    l = len(l_gaze_res.saccade_points)/len(l_gaze)
    r_gaze_res = saccademodel.fit(r_gaze)
    r = len(r_gaze_res.saccade_points)/len(r_gaze)
    return ((l+r)/2)

def computeGlabellarLength(glab_list):
    global GLAB_THR
    glab_list = np.array(glab_list)
    glab_stress = glab_list[glab_list< GLAB_THR]
    return (len(glab_stress)/len(glab_list))

def getReference():
    
    global EAR_THR 
    global OPEN_EYE 
    global GLAB_THR 
    global IRIS_SIZE_PX 
    global FOCAL
    print(EAR_THR,OPEN_EYE,GLAB_THR,IRIS_SIZE_PX,FOCAL)
    ratio_list = []
    frame_counter = 0
    with map_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        # starting time here 
        start_time = time.time()
        # starting Video loop here.
        while True:
             # frame counter
            frame_counter = frame_counter + 1
            ret, frame = camera.read() # getting frame from camera 
            if not ret: 
                break # no more frames break
            
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                mesh_coords=np.array([np.multiply([p.x, p.y], [frame_width, frame_height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                ratio_list.append(ratio)
                EAR_THR = min(ratio_list)
                OPEN_EYE = max(ratio_list)
                GLAB_THR = glabLength(mesh_coords, GLABELLAR)
                IRIS_SIZE_PX = irisSize(mesh_coords, LEFT_IRIS, RIGHT_IRIS)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_coords[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_coords[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
               
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
                
                for p in GLABELLAR:
                    cv.circle(frame, mesh_coords[p] ,2,(255,0,0),-1)
                

                
            # calculating  frame per seconds FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time
            frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            frame =utils.textWithBackground(frame,f'Getting REFERENCE Values, Relax and blink once',FONTS, 1.0, (30, 70), bgOpacity=0.9, textThickness=2)
            # writing image for thumbnail drawing shape
            # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
            cv.imshow('frame', frame)
            end=time.time()-start_time
            if(end > 5):
                break
            key = cv.waitKey(2)
            if key==ord('q') or key ==ord('Q'):
                break
    return (EAR_THR, OPEN_EYE, GLAB_THR, IRIS_SIZE_PX)

def getRealtime(value):
    global CEF_COUNTER
    global TOTAL_BLINKS
    global EAR_THR

    frame_counter = 0
    ratio_list=[]
    iris_list=[]
    glab_list =[]
    l_gaze_list=[]
    r_gaze_list =[]
    with map_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        # starting time here 
        start_time = time.time()
        # starting Video loop here.
        while True:
             # frame counter
            frame_counter = frame_counter + 1
            ret, frame = camera.read() # getting frame from camera 
            if not ret: 
                break # no more frames break
            #  resizing frame
            
            
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
            
                mesh_coords=np.array([np.multiply([p.x, p.y], [frame_width, frame_height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE) # EAR
                iris = irisSize(mesh_coords,LEFT_IRIS,RIGHT_IRIS)           # Eye device Distance
                glab = glabLength(mesh_coords,GLABELLAR)                    # glabellar distance
            
                ratio_list.append(ratio)
                iris_list.append(iris)
                glab_list.append(glab)

                for p in GLABELLAR:
                    cv.circle(frame, mesh_coords[p] ,2,(255,0,0),-1)

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_coords[LEFT_IRIS])
                (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_coords[RIGHT_IRIS])
                
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
               
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
                # locus for eye movement duration
                l_eye =mesh_coords[LEFT_EYE[0]]
                r_eye = mesh_coords[RIGHT_EYE[0]]
                l_eye_gaze = [abs(l_cx -l_eye[0]), abs(l_cy - l_eye[1])]
                r_eye_gaze = [abs(r_cx -r_eye[0]), abs(r_cy - r_eye[1])]
            
                l_gaze_list.append(l_eye_gaze)
                r_gaze_list.append(r_eye_gaze)

                if ratio < EAR_THR:
                    CEF_COUNTER +=1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            # calculating  frame per seconds FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time
            frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 30), bgOpacity=0.9, textThickness=2)
            frame =utils.textWithBackground(frame,f'Eye Strain Test in Progress',FONTS, 1.0, (30, 70), bgOpacity=0.9, textThickness=2)
            cv.imshow('frame', frame)
            end=time.time()-start_time
            if end > value*60 :
                break
            key = cv.waitKey(2)
            if key==ord('q') or key ==ord('Q'):
                break
    blr = computeBlinkRate(end)
    Sed = computeSquintEyeDuration(ratio_list)
    Edd = computeEyeDeviceDistance(iris_list)
    #Emd = computeEyeMovementDuration(l_gaze_list, r_gaze_list)
    Gll = computeGlabellarLength(glab_list)   
    return blr, Sed, Edd, Gll

print(getRealtime(3))

