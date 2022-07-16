#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL

import time
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils import image2array, image_file_to_tensor

# Topic from which take the caemera raw images
RAW_TOPIC = '/image_raw'
# Topic on which publish the matted images
MATTED_TOPIC = '/image_raw_matted'

OUTPUTS_DIR = os.path.join('..', '..', 'outputs')
TEST_IMAGES_DIR = 'test_images'
BACKBONE = 'resnet50'

# Publisher for matted images
pub = rospy.Publisher(MATTED_TOPIC, Image, queue_size=1000) 
# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Initializing BackgroundMatting model
from background_matting import BackgroundMatting
model = BackgroundMatting(backbone=BACKBONE, input_resolution='nhd')

# Entering into Background Matting working directory
os.chdir('BackgroundMatting')
print("Current working directory: ", os.getcwd())
from BackgroundMatting.utils.model_utils import get_dummy_inputs

	
def callback(image):
	print("---------------------------- RAW IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the raw image")
	
	# Converting image message to Numpy RGB array
	rgb_array = image2array(image)
	# Saving image in outputs folder
	path_to_image_raw = os.path.join(OUTPUTS_DIR, 'image_raw.png')
	PIL.Image.fromarray(rgb_array).save(path_to_image_raw)

	# Getting inputs
	# Get the background image from another publisher or from a directory
	bgr = image_file_to_tensor(os.path.join(TEST_IMAGES_DIR, "background.png"))
	# The following input is ONLY FOR BACKGROUND MATTING TESTING!
	# src = image_file_to_tensor(os.path.join(TEST_IMAGES_DIR, 'dirty_0.png'))
	# The following input is for PRODUCTION USE
	src = image_file_to_tensor(path_to_image_raw)
	
	# Inference
	print("- Starting the background matting ...")
	start_time = time.time()
	tfgr = model.matting(src=src, bgr=bgr)
	inference_seconds = time.time() - start_time
	print(f"Matting took up {inference_seconds} seconds!")

	# Saving matted image
	print("- Saving matted image to ", os.path.join(OUTPUTS_DIR, 'image_raw_matted.png'))
	to_pil_image(tfgr.squeeze(0)).save(os.path.join(OUTPUTS_DIR, 'image_raw_matted.png'))

	try:
		# Reading and publishing matted image
		print("----- PUBLISHING MATTED IMAGE -----")
		path_to_matted = os.path.join(OUTPUTS_DIR, 'image_raw_matted.png')
		matted_cv = cv.imread(path_to_matted, flags=cv.IMREAD_UNCHANGED)
		
		# cv.imshow('RGBA image to publish', matted_cv)
		# cv.waitKey(0) 
		# cv.destroyAllWindows()

		msg = bridge.cv2_to_imgmsg(matted_cv, encoding="passthrough")
		pub.publish(msg)
		print("Matted image published successfully on ", MATTED_TOPIC)
		print("-----------------------------------")
	except Exception as e:
		print("WARNING! Can't publish the matted image.\nException: ", e)

	print("----------------------------------------------------------------")
	# Waiting for inputs
	input("> Press enter to keep on listening and publishing ...")

def listener():
	# In ROS, nodes are uniquely named. If two nodes with the same
	# name are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
  	# name for our 'listener' node so that multiple listeners can
  	# run simultaneously.
	rospy.init_node('camera_listener', anonymous=True)
	rospy.Subscriber(RAW_TOPIC, Image, callback)
	
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	# Preparing outputs directory
	if not os.path.exists(OUTPUTS_DIR):
		os.mkdir(OUTPUTS_DIR)
	print(f"- Starting listening to {RAW_TOPIC} ...")
	# Listening to raw images
	listener()
