import face_recognition
import cv2
import put_chinese_text

IMG_DIR1 = 'obama.jpg'
IMG_DIR2 = 'biden.jpg'
IMG_DIR3 = 'obama_and_biden.jpg'

obama_image = face_recognition.load_image_file(IMG_DIR1)    #obama
biden_image = face_recognition.load_image_file(IMG_DIR2)    #biden
unknown_image = face_recognition.load_image_file(IMG_DIR3)  #obama
unknown_image_cv2 = cv2.imread(IMG_DIR3)

obama_encoding = face_recognition.face_encodings(obama_image)[0]    #只有一张人脸，取索引0
biden_encoding = face_recognition.face_encodings(biden_image)[0]    #只有一张人脸，取索引0

#已知人脸的encodings
known_image_encodings = [
    obama_encoding, 
    biden_encoding
    ]
#对应encoding的名字
known_faces = [
    '奥巴马', 
    '拜登'
    ]
#检测人脸，提取encodings
face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model="cnn")
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

face_names = []

#为每一个encoding对比已知人脸的encodings
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_image_encodings, face_encoding)    #对第一个参数中每张人脸的True/False列表
    name = 'unknown'

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]

    for i in matchedIdxs:
        name = known_faces[i]
    face_names.append(name)
#画出每一张人脸及其名字
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(unknown_image_cv2, (left, top), (right, bottom), (0, 255, 0), 2)
    top_text = top - 25 if top - 25 > 25 else top + 25
    cv2.rectangle(unknown_image_cv2, (left, top_text), (right, top), (0, 255, 0), cv2.FILLED)
    unknown_image_cv2 = put_chinese_text.draw_text(unknown_image_cv2, name, (left, top_text), (0, 0, 0))

    # font = cv2.FONT_HERSHEY_COMPLEX
    # cv2.putText(unknown_image_cv2, name, (left, top_text), font, 1, (0, 0, 0), 1)

cv2.imshow('faces', unknown_image_cv2)
cv2.waitKey(0)