#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL

import time
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils import image2array, image_file_to_tensor

# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Topic from which take the matted images
MATTED_TOPIC = '/image_raw_matted'

OUTPUTS_DIR = os.path.join('..', 'outputs')

def callback(image):
	print("---------------------------- MATTED IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the matted image")
	
	# Converting image message to Numpy RGBA array
	# rgba_array = image2array(image)
	rgba_cv = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	
	
	# TODO: Get the alpha channel
	# Waiting for inputs
	pass
	input("> Press any key to keep on listening and publishing ...")
    
	try:
		# Reading and publishing dirty coordinates
		print("----- PUBLISHING DIRTY COORDINATES -----")
		
		# pub.publish(msg)
		# print("Matted image published successfully on ", MATTED_TOPIC)
		print("-----------------------------------")
	except Exception as e:
		print("WARNING! Can't publish the dirty coordinates.\nException: ", e)

	print("----------------------------------------------------------------")
	

def listener():
	rospy.init_node('matted_listener', anonymous=True)

	print(f"- Starting listening to {MATTED_TOPIC} ...")
	rospy.Subscriber(MATTED_TOPIC, Image, callback)
	
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	# Listening to matted images
	listener()
