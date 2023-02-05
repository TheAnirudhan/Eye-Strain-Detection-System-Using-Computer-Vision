import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time
mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

#indices da iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

L_H_LEFT = [33]     # olho direito ponto de referência mais à direita
L_H_RIGHT = [133]   # olho direito ponto de referência mais à esquerda
R_H_LEFT = [362]    # olho esquerdo ponto de referência mais à direita
R_H_RIGHT = [263]   # olho esquerdo ponto de referência mais à esquerda
iris_list=[]

#Funcao que define distancias euclidianas dos pontos nos olhos
def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

#Funcao para encontra posicao da iris
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""

    if ratio <= 0.42:
        iris_position="right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
    else:
        iris_position = "left"
    return iris_position, ratio
start=time.time()
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   #Mediapipe precisa do formato de cores RGB mas o OpenCV usa o BGR
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            #transformar formas quadradas em círculos, função do OpenCV fornece círculos delimitadores com base nos pontos fornecidos.
            #minEnclosingCircle que retorna, o centro (x,y) e o raio dos círculos, os valores de retorno são de ponto flutuante, necessario transformá-los em int.
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # transforma pontos centrais em array np
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            #desenhe o círculo com base nos valores de retorno da minEnclosingCircle, através do CIRCLE que desenha a imagem do círculo com base no centro (x, y) e no raio
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            #mostrar pontos nos cantos dos olhos
            st=(mesh_points[R_H_RIGHT[0]][0]+30,mesh_points[R_H_RIGHT][0][1]-20)
            end=(mesh_points[R_H_LEFT][0][0]-20,mesh_points[R_H_LEFT][0][1]+20)
            cv.rectangle(frame,st,end,(255,0,0),1)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            iris_list.append(center_right)
        cv.imshow("img", frame)
        key = cv.waitKey(1)
        if key ==ord("q"):
            break
tot_dist=0
for i  in range(len(iris_list)-1):
    a=euclidean_distance(iris_list[i],iris_list[i+1])
    tot_dist=tot_dist+a
end_time=time.time()-start
eye_mov_vel=tot_dist/end_time
eyem_flag=''
if int(eye_mov_vel) in range(20,35):
    eyem_flag='optimal'
elif eye_mov_vel <20:
    eyem_flag='sub optimal'
else:
    eyem_flag='non optimal'
print(f"Eye Movement Velocity : {str(round(eye_mov_vel,2))} units per second | Optimality : {eyem_flag}")
cap.release()
cv.destroyAllWindows()