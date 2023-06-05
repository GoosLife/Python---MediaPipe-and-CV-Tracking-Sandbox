import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh

with mpFaceMesh.FaceMesh(
    max_num_faces = 2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        faceResults = face_mesh.process(image)

        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

        # checking whether a face is detected
        if faceResults.multi_face_landmarks:
            for face_landmarks in faceResults.multi_face_landmarks:
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
        cv2.imshow("Output", cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()