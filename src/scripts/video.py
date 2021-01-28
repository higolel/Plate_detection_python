#!/usr/bin/env python3
#!coding=utf-8

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sys
import tensorflow as tf
import time
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import HyperLPRLite as pr

tf_config = tf.compat.v1.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 分配40%
tf_config.gpu_options.allow_growth = True # 自适应
session = tf.compat.v1.Session(config = tf_config)


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
# \概要:    web视频显示
#
# \返回:
# --------------------------------------------------------
def webcamImagePub():
    # init ros_node
    rospy.init_node('plate_detection', anonymous = True)
    model_ = rospy.get_param('model')
    font_ = rospy.get_param('font')
    video_ = rospy.get_param('~video')
    video_sub_ = 'H.265/ch1/main/av_stream'
    video_ = video_ + video_sub_

    fontC = ImageFont.truetype(font_, 14, 0)
    # queue_size should be small in order to make it 'real_time'
    # or the node will pub the past_frame
    #from sensor_msgs.msg import Image
    #img_pub = rospy.Publisher('webcam/image_raw', Image, queue_size = 2)
    rate = rospy.Rate(5) # 5hz
    # model load
    model = pr.LPR(model_ + "cascade.xml", model_ + "model12.h5", model_ + "ocr_plate_all_gru.h5")

    # make a video_object and init the video object
    cap = cv2.VideoCapture(video_)
    # define picture to_down' coefficient of ratio
    scaling_factor = 0.5
    # the 'CVBridge' is a python_class, must have a instance.
    # That means "cv2_to_imgmsg() must be called with CvBridge instance"
    bridge = CvBridge()

    if not cap.isOpened():
        sys.stdout.write("Webcam is not available !")
        return -1

    count = 0
    # loop until press 'esc' or 'q'
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        # resize the frame
        if ret:
            count = count + 1
        else:
            rospy.loginfo("Capturing image failed.")
        if count >= 2:
            count = 0
            for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(frame):
                if confidence > 0.7:
                    frame = drawRectBox(frame, rect, pstr + " " + str(round(confidence, 3)), fontC)
                    print("plate_str:")
                    print(pstr)
                    print("plate_confidence")
                    print(confidence)

            frame = cv2.resize(frame, None, fx = scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_AREA)
            #msg = bridge.cv2_to_imgmsg(frame, encoding = "bgr8")
            #img_pub.publish(msg)
            print('** publishing webcam_frame ***')
        rate.sleep()

if __name__ == '__main__':
    try:
        webcamImagePub()
    except rospy.ROSInterruptException:
        pass
