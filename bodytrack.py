import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture from the webcam
cap = cv2.VideoCapture(2)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    # Read a frame from the webcam
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue



    # Convert the BGR image (OpenCV default) to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find pose landmarks
    results = pose.process(image_rgb)




    # Draw the pose annotation on the BGR image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Customize landmark appearance
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            # Customize connection appearance
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # Display the resulting frame
    cv2.imshow('MediaPipe Body Tracking', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()