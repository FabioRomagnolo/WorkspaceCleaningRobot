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
from std_srvs.srv import Empty

# Topic from which take the caemera raw images
RAW_TOPIC = '/image_raw'
# Topic on which publish the matted images
MATTED_TOPIC = '/image_raw_matted'

OUTPUTS_DIR = os.path.join('..', '..', 'outputs')
TEST_IMAGES_DIR = 'test_images'

# Publisher for matted images
pub = rospy.Publisher(MATTED_TOPIC, Image, queue_size=1000) 
# OpenCV bridge for image messages
from cv_bridge import CvBridge
bridge = CvBridge()

# Initializing BackgroundMatting model -> Set model params here!
from background_matting import BackgroundMatting
model = BackgroundMatting(backbone='resnet50', refine_mode='full', input_resolution='nhd')

# Entering into Background Matting working directory
os.chdir('BackgroundMatting')
print("Current working directory: ", os.getcwd())
from BackgroundMatting.utils.model_utils import get_dummy_inputs

service_name = 'background_matting'


def handle_request(data):
	print('Request received, starting background matting...')
	print('Listening to the most recent image from the camera...')
	image_message = rospy.wait_for_message(RAW_TOPIC, Image)
	print("---------------------------- RAW IMAGE LISTENED! ----------------------------")
	rospy.loginfo(rospy.get_caller_id() + "I heard the raw image")
	
	# Converting image message to Numpy RGB array
	rgb_array = image2array(image_message)
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
	#input("> Press enter to keep on listening and publishing ...")

if __name__ == '__main__':
	# Preparing outputs directory
	if not os.path.exists(OUTPUTS_DIR):
		os.mkdir(OUTPUTS_DIR)

	rospy.init_node('background_matting', anonymous=True)
	print(f'Starting {service_name} service')
	s = rospy.Service(service_name, Empty, handle_request)
	print('Service started. Ready to handle requests')
	rospy.spin()


