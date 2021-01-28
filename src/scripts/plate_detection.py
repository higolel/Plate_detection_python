#!/usr/bin/env python3
#!coding=utf-8

import cv2
import numpy as np
#from cv_bridge import CvBridge, CvBridgeError
import sys
import datetime
import tensorflow as tf
import time
import base64
import queue
import threading
import rospy
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from face_plate_msgs.msg import Plate_pic
from location.msg import location, gps
import HyperLPRLite as pr

tf_config = tf.compat.v1.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 分配40%
tf_config.gpu_options.allow_growth = True # 自适应
session = tf.compat.v1.Session(config = tf_config)


lon_ = 0.0
lat_ = 0.0
que = queue.Queue()
# --------------------------------------------------------
##
# \概要:    经纬度回调函数
#
# \参数:    msg
#
# \返回:
# --------------------------------------------------------
def locationMsgCallback(msg):
    location_msg = location()
    location_msg = msg
    lon_ = msg.gps.lon
    lat_ = msg.gps.lat

# --------------------------------------------------------
##
# \概要:    画框写文字
#
# \参数:    image
# \参数:    rect
# \参数:    addText
#
# \返回:    image
# --------------------------------------------------------
def drawRectBox(image, rect, addText, fontC):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode("utf-8").decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

# --------------------------------------------------------
##
# \概要:    制作文字图像
#
# \参数:    frame
# \参数:    fontC
#
# \返回:
# --------------------------------------------------------
def makeTextPic(frame, fontC):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    addText = "地点名称: " + "longitude: " + str(round(lon_, 6)) + "latitude: " + str(round(lat_, 6))
    draw.text((10, 40), addText.encode("utf-8").decode("utf-8"), (0, 0, 255), font = fontC)

    now = datetime.datetime.now()
    addText = "经过时间: " + now.strftime("%Y/%m/%d %H:%M:%S")
    draw.text((10, 100), addText.encode("utf-8").decode("utf-8"), (0, 0, 255), font = fontC)

    addText = "设备编号: " + "as00031"
    draw.text((10, 160), addText.encode("utf-8").decode("utf-8"), (0, 0, 255), font = fontC)
    imagex = np.array(img)

    return imagex

# --------------------------------------------------------
##
# \概要:    图片切割
#
# \参数:    frame
#
# \返回:
# --------------------------------------------------------
def imageIntercept(frame):
    sp = frame.shape
    height = sp[0]
    weight = sp[1]

    rect_frame = frame[int(height / 3) : int(height / 3 * 2), int(weight / 7) : int(weight / 7 * 5)]

    return rect_frame

def getPlateImage(frame, rect):
    rect_frame = frame[int(rect[1]) : int(rect[1] + rect[3]), int(rect[0]) : int(rect[0] + rect[2])]

    return rect_frame

def pubPlatePicMsg(pstr, plateImage, frame, plate_pub_):
    plate_pic_msg = Plate_pic()
    plate_pic_msg.vin = "as00030";
    plate_pic_msg.deviceId = "030车牌";
    plate_pic_msg.pictureType = 2;
    plate_pic_msg.lon = lon_;
    plate_pic_msg.lat = lat_;
    plate_pic_msg.licenseNum = pstr;
    plate_pic_msg.plateColor = 0;
    plate_pic_msg.carColor = 0;
    plate_pic_msg.carType = 0;

    t = time.time()
    plate_pic_msg.capTime = int(round(t * 1000))
    plate_pic_msg.licensePlatePicture = str(base64.b64decode(plateImage))
    plate_pic_msg.licensePlateScenePicture = str(base64.b64encode(frame))

    plate_pub_.publish(plate_pic_msg)
    print("pub success")

# --------------------------------------------------------
##
# \概要:    接收线程
#
# \参数:    video_
# \参数:    rate
#
# \返回:
# --------------------------------------------------------
def Receive(video_, rate_r):
    print('Start Receive')
    cap = cv2.VideoCapture(video_)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            print("receive success")
            que.put(frame)
            # Throw out the old image in the queue
            if que.qsize() > 1:
                que.get()
        else:
            rospy.loginfo("Capturing image failed.")

        rate_r.sleep()

# --------------------------------------------------------
##
# \概要:    发送线程
#
# \参数:    fontC_
# \参数:    plate_pub_
# \参数:    model_
# \参数:    rate
#
# \返回:
# --------------------------------------------------------
def Display(fontC_, plate_pub_, model_path, rate_d):
    print('Start Display')

    # model load
    model_ = pr.LPR(model_path + "cascade.xml", model_path + "model12.h5", model_path + "ocr_plate_all_gru.h5")
    while not rospy.is_shutdown():
        if que.empty() != True:
            frame = que.get()
            frame_intercept = imageIntercept(frame)
            for pstr, confidence, rect in model_.SimpleRecognizePlateByE2E(frame_intercept):
                print(confidence)
                if confidence > 0.9:
                    #frame = drawRectBox(frame_intercept, rect, pstr + " " + str(round(confidence, 3)), fontC)
                    print("plate_str:")
                    print(pstr)
                    print("plate_confidence")
                    print(confidence)
                    print(len(pstr))
                    if len(pstr) >= 7:
                        frame = makeTextPic(frame, fontC_)
                        plateImage = getPlateImage(frame_intercept, rect)
                        cv2.imwrite("~/plate.jpg", plateImage)
                        pubPlatePicMsg(pstr, plateImage, frame, plate_pub_)

            """
            frame = cv2.resize(frame, None, fx = scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_AREA)
            msg = bridge.cv2_to_imgmsg(frame, encoding = "bgr8")
            img_pub.publish(msg)
            """
            print('** publishing webcam_frame ***')
        else:
            print("que is empty")

        rate_d.sleep()



# --------------------------------------------------------
##
# \概要:   main
#
# \返回:
# --------------------------------------------------------
def main():
    # init ros_node
    rospy.init_node('plate_detection', anonymous = True)
    model_path = rospy.get_param('model')
    font = rospy.get_param('font')
    video = rospy.get_param('~video')
    video_sub = 'H.265/ch1/main/av_stream'
    video_ = video + video_sub

    fontC_ = ImageFont.truetype(font, 14, 0)
    """
    img_pub = rospy.Publisher('webcam/image_raw', Image, queue_size=2)
    """

    plate_pub_ = rospy.Publisher("/plate_pic_msg", Plate_pic, queue_size = 1)
    rospy.Subscriber('/location', location, locationMsgCallback)

    rate_r = rospy.Rate(15) # 5hz
    rate_d = rospy.Rate(1) # 5hz


    pthread_1 = threading.Thread(target = Receive, args = (video_, rate_r))
    pthread_2 = threading.Thread(target = Display, args = (fontC_, plate_pub_, model_path, rate_d))
    pthread_1.start()
    pthread_2.start()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
