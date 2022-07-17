#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt32MultiArray, Float32MultiArray
import tf2_ros
import tf

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from utils import transform_pixels2camera, transform_points

# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Topic from which take the matted images
MATTED_TOPIC = '/image_raw_matted'

# Topic on which publish dirty coordinates
DIRTY2D_TOPIC = '/dirty2d'
DIRTY3D_IMG2CAMERA_TOPIC = '/dirty2camera'
DIRTY3D_IMG2WORLD_TOPIC = '/dirty2world'

pub_dirty2d = rospy.Publisher(DIRTY2D_TOPIC, UInt32MultiArray, queue_size=10)
pub_dirty3d_img2camera = rospy.Publisher(DIRTY3D_IMG2CAMERA_TOPIC, Float32MultiArray, queue_size=10)
pub_dirty3d_img2world = rospy.Publisher(DIRTY3D_IMG2WORLD_TOPIC, Float32MultiArray, queue_size=10)

OUTPUTS_DIR = os.path.join('..', 'outputs')

def callback(image):
	print("---------------------------- MATTED IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the matted image")
	
	# Converting image message to Numpy RGBA array
	# rgba_array = image2array(image, image_type='rgba')
	bgra_cv = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	
	alpha = bgra_cv[:,:,-1]
	path_to_alpha = os.path.join(OUTPUTS_DIR, 'alpha_image_raw_matted.jpg')
	print("- Saving alpha channel of matted image to ", path_to_alpha)
	cv.imwrite(path_to_alpha, alpha)

	# 2D coordinates where we need to clean up. 
	# The array has shape (h, w) while we need x, y coordinates,so we have to swap axes
	dirty_2d = np.argwhere((alpha != 0)).astype('int32')
	dirty_2d[:, [1, 0]] = dirty_2d[:, [0, 1]]
	print(f"2D Coordinates of the dirty in the matted image {dirty_2d.shape}:\n{dirty_2d}")
	
	# Getting camera info
	camera_info = rospy.wait_for_message('/camera_info', CameraInfo)
	depth = 1.55 - 0.1  # Distance of camera from table: height(camera) - height(table)

	# Transforming 2D dirty pixels to 3D coordinates w.r.t. camera
	dirty2camera = transform_pixels2camera(pixels=dirty_2d, camera_info=camera_info, depth=depth)
	print(f"3D coordinates of pixels w.r.t. camera frame ({dirty2camera.shape}):\n", dirty2camera)

	# Transforming 3D coordinates w.r.t. world
	try:
		# Once the transform listener is created, it starts receiving tf2 transformations over the wire, 
		# and buffers them for up to 10 seconds
		tf2_buffer = tf2_ros.Buffer()
		tf2_listener = tf2_ros.TransformListener(tf2_buffer)

		# Getting camera w.r.t. world as TransformStamped.transform
		tf2_camera2world =  tf2_buffer.lookup_transform(
			'camera_link', 'world', time=rospy.Time(0), timeout=rospy.Duration(10)).transform

		print("- Getting transformation of camera frame w.r.t. world...\n", tf2_camera2world)
		translation, quaternions = tf2_camera2world.translation, tf2_camera2world.rotation

		# Transforming 3D coordinates w.r.t. world: tf implementation
		dirty2world = transform_points(points=dirty2camera, ref_frame='world', init_frame='camera_link')
		print(f"3D coordinates of pixels w.r.t. world frame ({dirty2world.shape}):\n", dirty2world)

	except Exception as e:
		print("WARNING! Can't compute the transformation of dirty w.r.t. world!\n", e)
	

	try:
		# Reading and publishing dirty coordinates
		print("----- PUBLISHING DIRTY COORDINATES -----")	
		
		# Publishing 2D coordinates
		msg = UInt32MultiArray()
		msg.data = dirty_2d.tolist()
		pub_dirty2d.publish(msg)
		print("Dirty 2D coordinates published successfully on ", DIRTY2D_TOPIC)

		# Publishing 3D coordinates w.r.t. camera
		msg = Float32MultiArray()
		msg.data = dirty2camera.tolist()
		pub_dirty3d_img2camera.publish(msg)
		print("Dirty 3D coordinates w.r.t. camera frame successfully published on ", DIRTY3D_IMG2CAMERA_TOPIC)

		# Publishing 3D coordinates w.r.t. world
		msg = Float32MultiArray()
		msg.data = dirty2world.tolist()
		pub_dirty3d_img2world.publish(msg)
		print("Dirty 3D coordinates w.r.t. camera frame successfully published on ", DIRTY3D_IMG2WORLD_TOPIC)


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
