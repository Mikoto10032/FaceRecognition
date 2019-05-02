import face_recognition
import cv2

IMG_DIR = 'example_01.png'

image = face_recognition.load_image_file(IMG_DIR)   #加载图像
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn") #每一张人脸的位置（top, right, bottom, left）

img = cv2.imread(IMG_DIR)

for i, bbox in enumerate(face_locations):
    print(bbox)
    top, right, bottom, left = bbox
    face = img[top:bottom, left:right]  #切割人脸
    cv2.imshow('face', face)
    # cv2.imwrite('face_' + str(i)+ '.jpg', face)
    cv2.waitKey(0)  #esc退出