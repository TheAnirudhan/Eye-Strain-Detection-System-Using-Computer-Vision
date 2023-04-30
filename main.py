import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import saccademodel
import matplotlib.pyplot as plt
from strain_class import *

# global variables
TEST_DURATION = 3
RECCURENCE_INTERVAL = 20 

EAR_THR  = 0.22
OPEN_EYE = 0.5
GLAB_THR = 18
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
    '''
    Euclidean distance is a mathematical formula used to calculate the distance between two points in a two- or three-dimensional space. It is the straight-line distance between two points in a Cartesian coordinate system, and can be calculated using the Pythagorean theorem.

    '''
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def glabLength(landmarks, glab):
    l1 = cv.norm(landmarks[glab[0]],landmarks[glab[1]])
    l2 = cv.norm(landmarks[glab[2]],landmarks[glab[3]])
    l3 = cv.norm(landmarks[glab[4]],landmarks[glab[5]])
    return ((l1+l2+l3)/3)

def irisSize(landmarks, left_iris, right_iris):
    l_size = euclaideanDistance(landmarks[left_iris[0]], landmarks[left_iris[2]])
    r_size = euclaideanDistance(landmarks[right_iris[0]],landmarks[right_iris[2]])
    return ((l_size+ r_size) / 2)

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    '''
    From facial 68 landmark points the left and eye location
    points (36-48) are extracted.
    To measure a metric for opening and closing of eye we
    use a distance metric which is simply
    
    Eye Aspect Ratio = VERTICAL DISTANCE / HORIZONTAL DISTANCE

    When the EAR is below a threshold level and then returns
    back to previous level, the blink is being detected
    img: The input image on which the landmarks are detected.
    landmarks: A list of 68 facial landmarks detected in the input image. The landmarks are represented as (x, y) coordinates.
    right_indices: A list of indices of the facial landmarks corresponding to the right eye.
    left_indices: A list of indices of the facial landmarks corresponding to the left eye.
    rh_right: The landmark corresponding to the outer right corner of the right eye.
    rh_left: The landmark corresponding to the outer left corner of the right eye.
    rv_top: The landmark corresponding to the top midpoint of the right eye.
    rv_bottom: The landmark corresponding to the bottom midpoint of the right eye.
    lh_right: The landmark corresponding to the outer right corner of the left eye.
    lh_left: The landmark corresponding to the outer left corner of the left eye.
    lv_top: The landmark corresponding to the top midpoint of the left eye.
    lv_bottom: The landmark corresponding to the bottom midpoint of the left eye.
    rhDistance: The horizontal distance between the outer right and left corners of the right eye.
    rvDistance: The vertical distance between the top and bottom midpoints of the right eye.
    lvDistance: The vertical distance between the top and bottom midpoints of the left eye.
    lhDistance: The horizontal distance between the outer right and left corners of the left eye.
    reRatio: The eye aspect ratio (EAR) of the right eye, calculated as the vertical distance divided by the horizontal distance.
    leRatio: The EAR of the left eye, calculated as the vertical distance divided by the horizontal distance.
    ratio: The average of the EARs of the right and left eyes, used as a measure of the opening and closing of the eyes (blinking).
    '''
    # RIGHT_EYE 
    # horizontal 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # euclaideanDistance
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    
    reRatio = rvDistance/rhDistance # The eye aspect ratio (EAR) of the right eye

    # LEFT_EYE 
    # horizontal  
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical  
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    # euclaideanDistance
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    
    leRatio = lvDistance/lhDistance # The eye aspect ratio (EAR) of the left eye

    # draw lines on right eyes                                  # debug
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)


    ratio = (reRatio+leRatio)/2
    return ratio

# Feature 1
def computeBlinkRate(duration):
    global TOTAL_BLINKS
    return (TOTAL_BLINKS*60/duration)

# Feature 2
def computeSquintEyeDuration(ear):
    global OPEN_EYE
    ear=np.array(ear)
    squint_list = ear[ear < 0.6*OPEN_EYE ]
    return (len(squint_list)/len(ear)), squint_list

# Feature 3
def computeEyeDeviceDistance(iris_list):
    iris_list = np.array(iris_list)
    return (np.mean(FOCAL*(12/iris_list)+10)), FOCAL*(12/iris_list)+10

def computeEyeMovementDuration(l_gaze, r_gaze):
    '''
    This function computes the duration of eye movement by taking the gaze coordinates of the left and right eyes as input.

    First, the function converts the gaze coordinates of both eyes into numpy arrays. Then, it calculates the absolute difference between consecutive gaze points for both eyes.

    Next, it filters out the gaze points that have a difference less than or equal to 2 (which are considered to be noise). After that, it calculates the proportion of filtered gaze points to the total number of gaze points for each eye.

    Finally, it calculates the average of the proportion of gaze points that pass the filter for the left and right eyes and returns the result as the duration of eye movement.
    '''
    l_gaze=np.array(l_gaze)
    r_gaze=np.array(r_gaze)
    l = np.abs(l_gaze[0:-1] - l_gaze[1:])
    r = np.abs(r_gaze[0:-1] - r_gaze[1:])
    l = l[l>2]
    r = r[r>2]
    l = np.size(l)/len(l_gaze)
    r = np.size(l)/len(r_gaze)
    return ((l+r)/2)

def computeGlabellarLength(glab_list):
    global GLAB_THR
    glab_list = np.array(glab_list)
    glab_stress = glab_list[glab_list< GLAB_THR]
    return (len(glab_stress)/len(glab_list)),glab_stress

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
#ear,oe,gt,ir = getReference()
#print("Eye Ratio Threshold : %.2f \nOpen Eye Ratio : %.2f\nGlabellar Threshold : %d\n Iris Size in Pixel : %d"%(ear,oe,gt,ir))
def getRealtime(value):
    global CEF_COUNTER
    global TOTAL_BLINKS
    global EAR_THR

    frame_counter = 0
    blink_ratio_list=[]
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
                break # no more frames, break
            
            
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
            
                mesh_coords=np.array([np.multiply([p.x, p.y], [frame_width, frame_height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                blink_ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE) # EAR
                iris = irisSize(mesh_coords,LEFT_IRIS,RIGHT_IRIS)           # Eye device Distance
                glab = glabLength(mesh_coords,GLABELLAR)                    # glabellar distance
            
                blink_ratio_list.append(blink_ratio)
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

                if blink_ratio < EAR_THR:
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
                utils.colorBackgroundText(frame,  f'Ratio : {round(blink_ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
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
    Sed,sq_lst = computeSquintEyeDuration(blink_ratio_list)
    Edd,Edd_list = computeEyeDeviceDistance(iris_list)
    Emd = computeEyeMovementDuration(l_gaze_list, r_gaze_list)
    Gll, glab_stress = computeGlabellarLength(glab_list)   
    score,actual=strainScore(blr,Sed,Emd,Edd,Gll)
    print(score)
    y_pred = clf.predict([blr,Sed,Emd,Edd,Gll])
    if actual :
        print("Eye Strain Score : %d \nYour Eyes are Healthy"%(int(score)))
    else:
        print("Eye Strain Score : %d \nEyes Have Strain, Go relax for a bit, See nature"%(int(score)))    
    print("Random Forest Prediction Result %d"(y_pred))
    return blr,blink_ratio_list, Sed, sq_lst, Edd, Edd_list, Emd, Gll, glab_list, glab_stress





