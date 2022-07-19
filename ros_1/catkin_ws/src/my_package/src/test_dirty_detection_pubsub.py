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

from utils import transform_pixels2camera, transform_points, publish_numpy_array

from sklearn.cluster import MeanShift, estimate_bandwidth

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

import std_msgs.msg

cleaning_state_pub = rospy.Publisher('cleaning_state', std_msgs.msg.String, queue_size=10)

def callback(image):
	print("---------------------------- MATTED IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the matted image")

	# Publishing state
	cleaning_state_pub.publish('dirty detection')
	
	# Converting image message to Numpy RGBA array
	# rgba_array = image2array(image, image_type='rgba')
	bgra_cv = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	
	alpha = bgra_cv[:,:,-1]
	path_to_alpha = os.path.join(OUTPUTS_DIR, 'image_raw_matted_alpha.jpg')
	print("- Saving alpha channel of matted image to ", path_to_alpha)
	cv.imwrite(path_to_alpha, alpha)

	# 2D coordinates where we need to clean up. 
	# The array has shape (h, w) while we need x, y coordinates,so we have to swap axes
	dirty_2d = np.argwhere((alpha != 0)).astype('int32')
	dirty_2d[:, [1, 0]] = dirty_2d[:, [0, 1]]
	print(f"2D Coordinates of the dirty in the matted image {dirty_2d.shape}:\n{dirty_2d}")

	# Visualize the target points in the image
	image_message = rospy.wait_for_message('/image_raw', Image)
	cv_image_bgr = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
	cv_image_rgb = cv.cvtColor(cv_image_bgr, cv.COLOR_BGR2RGB)
	for p in dirty_2d:
		cv.circle(cv_image_rgb, (p[0], p[1]), 2, (0, 0, 255), 2)
	# plt.imshow(cv_image_rgb)
	# plt.show()
	# Save the target points in the image
	path_to_targets = os.path.join(OUTPUTS_DIR, 'dirty_targets.jpg')
	cv.imwrite(path_to_targets, cv_image_rgb)

	# Clustering
	if dirty_2d.shape[0] > 0:
		bandwidth = estimate_bandwidth(dirty_2d, quantile=0.2)
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		ms.fit(dirty_2d)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_.astype(np.int32)
		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)

		print("Number of estimated clusters : %d" % n_clusters_)
		print('Cluster centers:')
		print(cluster_centers)

		# Visualize the cluster centers
		image_message = rospy.wait_for_message('/image_raw', Image)
		cv_image_bgr = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
		cv_image_rgb = cv.cvtColor(cv_image_bgr, cv.COLOR_BGR2RGB)
		for p in cluster_centers:
			cv.circle(cv_image_rgb, (p[0], p[1]), 2, (0, 0, 255), 2)
		#plt.imshow(cv_image_rgb)
		#plt.show()
		cv.imwrite(os.path.join(OUTPUTS_DIR, 'dirty_clusters.jpg'), cv_image_rgb)
		dirty_2d = cluster_centers

	
	# Getting camera info
	camera_info = rospy.wait_for_message('/camera_info', CameraInfo)
	depth = 1.55 - 0.4  # Distance of camera from table: height(camera) - height(table)

	# Transforming 2D dirty pixels to 3D coordinates w.r.t. camera
	dirty2camera = transform_pixels2camera(pixels=dirty_2d, camera_info=camera_info, depth=depth)
	# Rounding coordinates and deleting duplicates
	dirty2camera = np.unique(np.round(dirty2camera, decimals=3), axis=0)
	print(f"Rounded 3D coordinates of pixels w.r.t. camera frame ({dirty2camera.shape}):\n", dirty2camera)

	# Transforming 3D coordinates w.r.t. world
	try:
		# Once the transform listener is created, it starts receiving tf2 transformations over the wire, 
		# and buffers them for up to 10 seconds
		tf2_buffer = tf2_ros.Buffer()
		tf2_listener = tf2_ros.TransformListener(tf2_buffer)

		# Getting camera w.r.t. world as TransformStamped.transform
		tf2_camera2world =  tf2_buffer.lookup_transform(
			'camera_link_optical', 'world', time=rospy.Time(0), timeout=rospy.Duration(10)).transform

		print("- Getting transformation of camera frame w.r.t. world...\n", tf2_camera2world)
		translation, quaternions = tf2_camera2world.translation, tf2_camera2world.rotation

		# Transforming 3D coordinates w.r.t. world: tf implementation
		dirty2world = transform_points(points=dirty2camera, target_frame='world', source_frame='camera_link_optical')
		# Rounding coordinates and deleting duplicates
		dirty2world = np.unique(np.round(dirty2world, decimals=3), axis=0)
		print(f"Rounded 3D coordinates of pixels w.r.t. world frame ({dirty2world.shape}):\n", dirty2world)

	except Exception as e:
		print("WARNING! Can't compute the transformation of dirty w.r.t. world!\n", e)
	

	try:
		# Reading and publishing dirty coordinates
		print("----- PUBLISHING DIRTY COORDINATES -----")	
		
		if np.size(dirty_2d):
			# Publishing 2D coordinates
			publish_numpy_array(dirty_2d, pub_dirty2d)
			print("Dirty 2D coordinates published successfully on ", DIRTY2D_TOPIC)
		else:
			# Publishing state
			cleaning_state_pub.publish('no dirt')
			print("WARNING: None dirty 2D coordinates to publish.")

		if np.size(dirty2camera):
			# Publishing 3D coordinates w.r.t. camera
			publish_numpy_array(dirty2camera, pub_dirty3d_img2camera)
			print("Dirty 3D coordinates w.r.t. camera frame successfully published on ", DIRTY3D_IMG2CAMERA_TOPIC)
		else:
			print("WARNING! None dirty 3D coordinates w.r.t. camera to publish.")

		if np.size(dirty2world):
			# Publishing 3D coordinates w.r.t. world
			publish_numpy_array(dirty2world, pub_dirty3d_img2world)
			print("Dirty 3D coordinates w.r.t. world frame successfully published on ", DIRTY3D_IMG2WORLD_TOPIC)
		else:
			print("WARNING! None dirty 3D coordinates w.r.t. world to publish.")

		print("-----------------------------------")
	except Exception as e:
		print("WARNING! Can't publish the dirty coordinates.\nException: ", e)

	# Waiting for inputs
	pass
	#input("> Press enter to keep on listening and publishing ...")
    
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
