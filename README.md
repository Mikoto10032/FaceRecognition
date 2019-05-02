# FaceRecognition

## 人脸识别项目调研

### 1. 人脸相关的任务

- **Face Detection**
- **Face Alignment**
- **Face Recognition && Face Identification && Face Verification && Face Representation** 
- **Face(Facial) Attribute && Face(Facial) Analysis**
- **Face Reconstruction**
- **Face Tracking**
- **Face Clustering**
- **Face Super-Resolution && Face Deblurring && Face Hallucination**
- **Face Generation && Face Synthesis && Face Completion && Face Restoration**
- **Face Transfer && Face Editing && Face swapping**
- **Face Anti-Spoofing**
- **Face Retrieval**

### 2. 数据集

| Datasets                   | Description                                                  | Links                                                        | Publish Time |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **CASIA-WebFace**          | **10,575** subjects and **494,414** images                   | [Download](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) | 2014         |
| **MegaFace**               | **1 million** faces, **690K** identities                     | [Download](http://megaface.cs.washington.edu/)               | 2016         |
| **MS-Celeb-1M**            | about **10M** images for **100K** celebrities Concrete measurement to evaluate the performance of recognizing one million celebrities | [Download](http://www.msceleb.org/)                          | 2016         |
| **LFW**                    | **13,000** images of faces collected from the web. Each face has been labeled with the name of the person pictured. **1680** of the people pictured have two or more distinct photos in the data set. | [Download](http://vis-www.cs.umass.edu/lfw/)                 | 2007         |
| **VGG Face2**              | The dataset contains **3.31 million** images of **9131** subjects (identities), with an average of 362.6 images for each subject. | [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)  | 2017         |
| **UMDFaces Dataset-image** | **367,888 face annotations** for **8,277 subjects.**         | [Download](http://www.umdfaces.io/)                          | 2016         |
| **Trillion Pairs**         | Train: **MS-Celeb-1M-v1c** & **Asian-Celeb** Test: **ELFW&DELFW** | [Download](http://trillionpairs.deepglint.com/overview)      | 2018         |
| **FaceScrub**              | It comprises a total of **106,863** face images of male and female **530**celebrities, with about **200 images per person**. | [Download](http://vintage.winklerbros.net/facescrub.html)    | 2014         |
| **Mut1ny**                 | head/face segmentation dataset contains over 17.3k labeled images | [Download](http://www.mut1ny.com/face-headsegmentation-dataset) | 2018         |
| **IMDB-Face**              | The dataset contains about 1.7 million faces, 59k identities, which is manually cleaned from 2.0 million raw images. | [Download](https://github.com/fwang91/IMDb-Face)             | 2018         |

### 3. 人脸识别

人脸识别问题宏观上分为两类：1. 人脸验证（又叫人脸比对）2. 人脸识别。

#### 3.1 人脸验证

人脸验证做的是 1 比 1 的比对，即判断两张图片里的人是否为同一人。**最常见的应用场景便是人脸解锁**，终端设备（如手机）只需将用户事先注册的照片与临场采集的照片做对比，判断是否为同一人，即可完成身份验证。

#### 3.2 [人脸识别](https://github.com/ChanChiChoi/awesome-Face_Recognition#face-recognition)

人脸识别做的是 1 比 N 的比对，即判断系统当前见到的人，为事先见过的众多人中的哪一个。比如**疑犯追踪，小区门禁，会场签到，以及新零售概念里的客户识别**。这些应用场景的共同特点是：人脸识别系统都**事先存储了大量的不同人脸和身份信息**，系统运行时需要将见到的人脸与之前存储的大量人脸做比对，找出匹配的人脸。

#### 3.3 人脸识别正确率

##### 3.3.1 [LFW](http://vis-www.cs.umass.edu/lfw/results.html#ctbc)数据集

![1556539143745](/home/haozanliang/.config/Typora/typora-user-images/1556539143745.png)

![1556539179482](/home/haozanliang/.config/Typora/typora-user-images/1556539179482.png)

![1556539250461](/home/haozanliang/.config/Typora/typora-user-images/1556539250461.png)

##### 3.3.2 MEGAFACE数据集

![1556539422332](/home/haozanliang/.config/Typora/typora-user-images/1556539422332.png)

##### 3.3.3 MS-CELEB-1M数据集

![1556539478579](/home/haozanliang/.config/Typora/typora-user-images/1556539478579.png)

##### 3.3.4 IJB-A数据集

![1556539512787](/home/haozanliang/.config/Typora/typora-user-images/1556539512787.png)

#### 3.4 极速人脸检测开源项目

- [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
- [DSFD: Dual Shot Face Detector](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)

#### 3.5 好用的人脸识别开源库

- [Face Recognition](https://github.com/ageitgey/face_recognition) && [使用实例1](https://zhuanlan.zhihu.com/p/26431250) && [使用实例2](https://zhuanlan.zhihu.com/p/33456076)
- [OpenFace](https://github.com/cmusatyalab/openface)

#### 3.6 经典模型之[face++](https://arxiv.org/pdf/1501.04690.pdf)

【Naive-Deep Face Recognition: Touching the Limit of LFW Benchmark or Not?】

##### 3.6.1性能

| 训练集                   | 测试集 | 正确率 |
| ------------------------ | ------ | ------ |
| 网上收集的5million张人脸 | LFW    | 0.9950 |

##### 3.6.2 问题一

在真实场景测试中（Chinese ID （CHID）），该系统的假阳性率（![FP=10^{-5} ](https://www.zhihu.com/equation?tex=FP%3D10%5E%7B-5%7D+)）非常低。但是，真阳性率仅为0.66，没有达到真实场景应用要求。其中，年龄差异（包括intra-variation：同一个人，不同年龄照片；以及inter-variation：不同人，不同年龄照片）是影响模型准确率原因之一。而在该测试标准(CHID)下，人类表现的准确率大于0.90。

##### 3.6.3 问题二

数据采集偏差。基于网络采集的人脸数据集存在偏差。这些偏差表现在：（1）个体之间照片数量差异很大；（2）大部分采集的照片都是：微笑，化妆，年轻，漂亮的图片。这些和真实场景中差异较大。因此，尽管系统在LFW数据集上有高准确率，在现实场景中准确率很低。

##### 3.6.4 问题三

模型测试加阳性率非常低，但是现实应用中，人们更关注真阳性率。

##### 3.6.5 问题四

人脸图片的角度，光线，闭合（开口、闭口）和年龄等差异相互的作用，导致人脸识别系统现实应用准确率很低。

### 4. face_recognition库的使用

#### 4.1 在Ubuntu16.04+上安装

##### 4.1.1 安装git

```bash
sudo apt-get install -y git
```

##### 4.1.2 安装cmake（Ubuntu16以上系统自带即可）

```bash
sudo apt-get install -y cmake
```

##### 4.1.3 安装python-pip

```bash
sudo apt-get install -y python-pip 
```

##### 4.1.4 安装无GPU支持的dlib

```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS
```

##### 4.1.5 安装有GPU支持的dlib（可选）

```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
```

貌似新版本默认就是--yes，无需添加可选参数

##### 4.1.6 安装face_recognition

```bash
pip install face_recognition
```

```bash
# 测试
face_recognition --help
```

#### 4.2 检测图像中的人脸

```python
def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
```

```Python
def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
```

```python
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
```

#### 4.3 识别图像中的人脸特征

```python
def face_landmarks(face_image, face_locations=None, model="large"):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
```

```python
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
```

#### 4.4 识别图像中的人脸

```python
def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
```

```
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
```

```python
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

FONT_DIR = 'NotoSansCJK-Black.ttc'

def draw_text(image, strs, local, color, font_dir=FONT_DIR):
    """利用PIL在图像上显示中文

    Parameters
        image: np array, bgr通道格式
        strs: str，显示的中文
        local: tuple，（x，y）
        sizes: int，字体大小
        color: tuple, （r, g, b）

    Return
        image: np array，bgr
    """
    size = np.floor(3e-2 * image.shape[0]).astype('int32')
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype(font_dir, size, encoding="utf-8")
    draw.text(local, strs, color, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image
```

```python
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
```

#### 4.5 识别视频文件中的人脸

```python
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
    cv2.putText(frame, fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
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
```

#### 4.6 识别摄像头中的人脸

```python
import cv2
import face_recognition
import put_chinese_text
import time

resize_ratio = 0.5

video_capture = cv2.VideoCapture(0)   # 读取摄像头传入的视频
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #视频帧数目

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
    ret, frame = video_capture.read()
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
    cv2.putText(frame, fps, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    # 显示
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```









