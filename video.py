#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()
# 调用摄像头
capture=cv2.VideoCapture('porn.mp4') # capture=cv2.VideoCapture("1.mp4")
c = 1
timeRate = 10  # 截取视频帧的时间间隔（这里是每隔10秒截取一帧）
fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 这个是获取视频帧率
    FPS = capture.get(5)
    if ref:
        frameRate = int(FPS) * timeRate
        if(c % frameRate == 0):
            # 读取某一帧
            ref,frame=capture.read()
            # frame = cv2.imread('000600.jpg')
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            frame = np.array(yolo.detect_image(frame))

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite('./predict/porn{}.jpg'.format(c),frame)
        c+=1
        # cv2.imshow("video",frame)

capture.release()
    # c= cv2.waitKey(30) & 0xff 
    # if c==27:
    #     capture.release()
    #     break
