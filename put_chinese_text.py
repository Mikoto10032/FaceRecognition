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