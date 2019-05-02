import cv2
import face_recognition
import put_chinese_text
import time

VIDEO_DIR = 'hamilton_clip.mp4'
resize_ratio = 0.5

input_video = cv2.VideoCapture(VIDEO_DIR)   # 读取视频文件
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)) #视频帧数

fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # 视频编码器
output_video = cv2.VideoWriter('hamilton_clip_recognition_face.mp4', fourcc, 29.97, (640, 360)) # 打开一个视频写入文件 

# encoding两张需要识别的人脸
lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

# 已知人脸encodings
known_face_encodings = [
    lmm_face_encoding,
    al_face_encoding
]

known_face_names = [
    'lin',
    'alex'
]

frame_number = 0

while True:
    start_time = time.time()
    # 读取视频流的每一帧
    ret, frame = input_video.read()
    # 放缩每一帧，加速
    small_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

    frame_number += 1
    # 视频结束
    if not ret:
        break
    
    # BGR2RGB，face_recognition库需要此格式
    rgb_frame = small_frame[:, :, ::-1]
    # 获取当前帧所有人脸及其encodings
    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for encoding in face_encodings:
        # 为每一个encoding对比已知的encodings
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = 'unknown'
        # 找出结果为True所对应的名字
        if True in matches:
            matchIds = [i for (i, b) in enumerate(matches) if b]
            name = known_face_names[matchIds[0]]
        face_names.append(name)

    # 标记人脸和名字
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 如果使用了放缩帧，从新计算原始坐标位置
        top, right, bottom, left = int(top / resize_ratio), int(right / resize_ratio), int(bottom / resize_ratio), int(left / resize_ratio)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        top_text = top - 16 if top - 16 > 16 else top + 16
        cv2.rectangle(frame, (left, top_text), (right, top), (0, 0, 255), cv2.FILLED)
        frame = put_chinese_text.draw_text(frame, name, (left, top_text), (0, 0, 0))
    end_time = time.time()
    fps = '%.2f FPS'% (1 / (end_time - start_time))
    # cv2.putText(frame, fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, fps, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    # 将该帧写入文件
    output_video.write(frame)
    # 显示
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

input_video.release()
cv2.destroyAllWindows()
if output_video is not None:
	output_video.release()