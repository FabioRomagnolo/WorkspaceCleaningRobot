#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt32MultiArray, Float32MultiArray
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from utils import image2array, image_file_to_tensor, transform_pixels2camera

# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Topic from which take the matted images
MATTED_TOPIC = '/image_raw_matted'

# Topic on which publish dirty coordinates
DIRTY2D_TOPIC = '/dirty2d'
DIRTY3D_IMG2CAMERA_TOPIC = '/dirty3d'
pub_dirty2d = rospy.Publisher(DIRTY2D_TOPIC, UInt32MultiArray, queue_size=10)
pub_dirty3d_img2camera = rospy.Publisher(DIRTY3D_IMG2CAMERA_TOPIC, Float32MultiArray, queue_size=10)

OUTPUTS_DIR = os.path.join('..', 'outputs')

def callback(image):
	print("---------------------------- MATTED IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the matted image")
	
	# Converting image message to Numpy RGBA array
	# rgba_array = image2array(image, image_type='rgba')
	bgra_cv = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	
	alpha = bgra_cv[:,:,-1]
	# 2D coordinates where we need to clean up
	dirty_2d = np.argwhere((alpha != 0)).astype('int32')
	print(f"2D Coordinates of the dirty in the matted image {dirty_2d.shape}:\n{dirty_2d}")
	
	# Getting camera info
	camera_info = rospy.wait_for_message('/camera_info', CameraInfo)
	d = 1.55 - 0.1  # Distance of camera from table: height(camera) - height(table)

	# Transforming 2D dirty pixels to 3D coordinates w.r.t. camera
	dirty2camera = transform_pixels2camera(pixels=dirty_2d, camera_info=camera_info, distance=d)
	print(f"3D coordinates of pixels w.r.t. camera frame ({dirty2camera.shape}):\n", dirty2camera)

	try:
		# Reading and publishing dirty coordinates
		print("----- PUBLISHING DIRTY COORDINATES -----")	
		
		# Publishing 2D coordinates
		msg = UInt32MultiArray()
		msg.data = dirty_2d.tolist()
		pub_dirty2d.publish(msg)
		print("Dirty 2D coordinates published successfully on ", DIRTY2D_TOPIC)

		# Publishing 3D coordinates
		msg = Float32MultiArray()
		msg.data = dirty2camera.tolist()
		pub_dirty3d_img2camera.publish(msg)
		print("Dirty 3D coordinates w.r.t. camera frame successfully published on ", DIRTY3D_IMG2CAMERA_TOPIC)


		print("-----------------------------------")
	except Exception as e:
		print("WARNING! Can't publish the dirty coordinates.\nException: ", e)

	# Waiting for inputs
	pass
	input("> Press enter to keep on listening and publishing ...")
    
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
