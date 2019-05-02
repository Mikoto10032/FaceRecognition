import face_recognition
import cv2
import numpy as np

IMG_DIR = 'example_01.png'

image = face_recognition.load_image_file(IMG_DIR)   #加载图像
face_landmarks_list = face_recognition.face_landmarks(image)    #每一张人脸的特征字典，字典键值为各种属性

facial_features = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
    ]

img = cv2.imread(IMG_DIR)

for i, face_landmarks in enumerate(face_landmarks_list):
    for facial_feature in facial_features:
        print(facial_feature, face_landmarks[facial_feature])
        pts = np.array(face_landmarks[facial_feature])
        cv2.polylines(img, [pts], False, (255, 255, 255), 5)    #画出人脸特征

cv2.imshow('face', img)
cv2.waitKey(0)  #esc退出