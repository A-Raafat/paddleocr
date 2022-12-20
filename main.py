#  OCR + Text Detection
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2

def run_ocr(ocr, img_path, recognition=True):
        
    if recognition :
        result = ocr.ocr(img_path, cls=True, rec=True)
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]

    im_show = image.copy()
    poly = Image.new('RGBA', image.size)
    pdraw = ImageDraw.Draw(poly)

    for bx in boxes:
        print(bx)
        [x1,y1],[x2,y2],[x3,y3],[x4,y4] = bx
        color1 = np.random.randint(0,255)
        color2 = np.random.randint(0,255)
        color3 = np.random.randint(0,255)
        cup_poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        pdraw.polygon(cup_poly,
                fill=(color1,color2,color3,100),outline=(0,0,0,255))

    im_show.paste(poly,mask=poly)

    return im_show, result



def grep_text_mapper(result):
    mapper={}
    for txt_cord, (txt_recog, txt_score) in result:
        mapper[str(txt_cord)] = txt_recog

    return mapper


def grep_textbox(mapper, image, pts) :
    listed_point = pts

    aaa=np.array(image).copy()
    pts=np.array(pts, dtype=int)

    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = aaa[y:y+h, x:x+w].copy()

    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

    Text = mapper[str(listed_point)]

    return dst2, Text
