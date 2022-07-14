#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32MultiArray
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from utils import image2array, image_file_to_tensor

# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Topic from which take the matted images
MATTED_TOPIC = '/image_raw_matted'

# Topic on which publish dirty 2D coordinates
DIRTY2D_TOPIC = '/dirty2d'
pub_dirty2d = rospy.Publisher(DIRTY2D_TOPIC, UInt32MultiArray, queue_size=10)

OUTPUTS_DIR = os.path.join('..', 'outputs')

def callback(image):
	print("---------------------------- MATTED IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the matted image")
	
	# Converting image message to Numpy RGBA array
	# rgba_array = image2array(image, image_type='rgba')
	bgra_cv = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	
	alpha = bgra_cv[:,:,-1]
	# Coordinates where alpha_channel are different from zero are the ones where we need to clean up
	non_zero_alpha = np.argwhere((alpha != 0)).astype('int32')

	print(f"Coordinates of the dirty in the matted image:\n{non_zero_alpha}")
	
	try:
		# Reading and publishing dirty coordinates
		print("----- PUBLISHING DIRTY COORDINATES -----")	
		msg = UInt32MultiArray()
		msg.data = non_zero_alpha.tolist()

		pub_dirty2d.publish(msg)
		print("Dirty 2D coordinates published successfully on ", MATTED_TOPIC)
		print("-----------------------------------")
	except Exception as e:
		print("WARNING! Can't publish the dirty coordinates.\nException: ", e)

	# Waiting for inputs
	pass
	input("> Press any key to keep on listening and publishing ...")
    
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
