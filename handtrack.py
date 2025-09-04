import cv2
import mediapipe as mp
import time
print("Hello World")

#video capture object
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
#you can hold control and right click a function to go to that function and know its parameters
hands = mpHands.Hands(static_image_mode=True)

mpDraw = mp.solutions.drawing_utils
#draws our hands on video output
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #extract information from results object
    #print(results.multi_hand_landmarks) it shows the coordinates of hands
    if(results.multi_hand_landmarks):
        for hand_landmarks in results.multi_hand_landmarks :
            for id, lm in enumerate(hand_landmarks.landmark):
                #print(id, lm)
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                #print(id,cx,cy)
                if(id == 12):
                    cv2.circle(img,(cx,cy),10,(0,255,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    #this displays the video
    cv2.imshow('Video', img)
    cv2.waitKey(1)
