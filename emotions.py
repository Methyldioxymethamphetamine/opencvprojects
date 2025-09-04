import cv2
import mediapipe as mp
import time
from deepface import DeepFace

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the Hand Tracking model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the Pose model
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start capturing video from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# FPS counter variables
pTime = 0
cTime = 0

# Start a loop to read frames from the webcam
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for all models
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    body_results = pose.process(image_rgb)

    # Draw face mesh annotations and perform emotion detection
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw face tessellation, contours, and irises
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # Find the bounding box for the face
            h, w, c = image.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Crop the face with some padding for emotion analysis
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cropped_face = image[y_min:y_max, x_min:x_max]

            # Perform emotion analysis using DeepFace
            if cropped_face.size > 0:
                try:
                    result = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False, silent=True)
                    dominant_emotion = result[0]['dominant_emotion']
                    # Display the emotion on the frame
                    cv2.putText(image, dominant_emotion, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    # In case DeepFace fails to detect a face, continue without error
                    pass

    # Draw hand landmarks and connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    # Draw body landmarks
    if body_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            body_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Display the final image
    cv2.imshow('MediaPipe Face, Hand, and Pose Tracking with Emotion Detection', image)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()