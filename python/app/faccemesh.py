import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face= facemesh.FaceMesh(static_image_mode=True,  min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = np.solutions.drawing_utils


while True:
  
    _, frm = cap.read()
    rgb = cv2.cvtColor( frm, cv2.COLOR_BGR2RGB)
    
    op =face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
          draw.draw_landmarks(
            frm, i, facemesh.FACE_CONNECTIONS, 
            landmark_drawing_spec=draw.DrawingSpec(circle_radius=1))
    
    
    cv2.imshow("window" ,frm)
    
    if cv2.waitKey(1) == 27:
      cap.release()
      cv2.destroyAllWindows()
      break