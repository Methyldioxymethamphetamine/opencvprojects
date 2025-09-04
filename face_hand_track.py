import cv2
import mediapipe as mp
import time

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


#fps counter variables
pTime = 0
cTime = 0

# Initialize the Face Mesh model
# 'max_num_faces' is set to 2 to detect up to two faces at once.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the Hand Tracking model
# 'max_num_hands' is set to 2 to detect up to two hands.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

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

# Start a loop to read frames from the webcam
while cap.isOpened():
    # Read a frame from the webcam
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, mark the image as not writeable to pass by reference.


    # MediaPipe requires RGB images, but OpenCV reads in BGR format.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for both face and hand detection
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    body_results = pose.process(image_rgb)
    # Make the image writeable again before drawing on it


    # Draw face mesh annotations
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw tessellation
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            # Draw contours
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            # Draw irises
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

    # Draw hand landmarks and connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    if body_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            body_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Customize landmark appearance
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            # Customize connection appearance
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )










    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    flipped_text_image = cv2.flip(image, 0)

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)



    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face and Hand Tracking', image )

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()