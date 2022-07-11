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

TMP_DIR = os.path.join('..', 'tmp')
BACKBONE = 'resnet50'

# Initializing BackgroundMatting model
from background_matting import BackgroundMatting
model = BackgroundMatting(backbone=BACKBONE, input_resolution='nhd')

# Entering into Background Matting working directory
os.chdir('BackgroundMatting')
print("Current working directory: ", os.getcwd())
from BackgroundMatting.utils.model_utils import get_dummy_inputs


def image2array(image):
	"""
	Method to convert image message to Numpy RGB array.
	"""
	#print(image.data)
	#print(image.width) # 640
	#print(image.height) # 480
	#print(len(image.data)) # 921600 = 640*480*3
	#print(image.encoding) # bgr8
	image_data = list(image.data)
	image_array = np.array(image_data).astype('uint8')
	bgr_image = image_array.reshape(image.height, image.width, 3)
	rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
	return rgb_image

	
def callback(image):
	rospy.loginfo(rospy.get_caller_id() + "I heard the raw image")
	
	# Converting image message to Numpy RGB array
	rgb_array = image2array(image)
	# Saving image in tmp folder
	PIL.Image.fromarray(rgb_array).save(os.path.join(TMP_DIR, 'image_raw.png'))
	# plt.imshow(rgb_array)
	# plt.savefig(os.path.join(TMP_DIR, 'image_raw'))
	# plt.show()

	# TODO: Get the background image from another publisher or from the TMP directory
	# The following lines are ONLY FOR BACKGROUND MATTING TESTING!
	src, bgr = get_dummy_inputs(resolution='nhd')
	
	# Inference
	print("- Starting the background matting ...")
	start_time = time.time()
	tfgr = model.matting(src=src, bgr=bgr)
	inference_seconds = time.time() - start_time
	print(f"Matting took up {inference_seconds}!")

	# Saving matted image
	print("- Saving matted image to ", os.path.join(TMP_DIR, 'matted_image_raw.png'))
	to_pil_image(tfgr.squeeze(0)).save(os.path.join(TMP_DIR, 'matted_image_raw.png'))

	# Waiting for inputs
	input("Press any key to continue listening images")
   
def listener():
	# In ROS, nodes are uniquely named. If two nodes with the same
	# name are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
  	# name for our 'listener' node so that multiple listeners can
  	# run simultaneously.
	rospy.init_node('camera_listener', anonymous=True)
	rospy.Subscriber("/image_raw", Image, callback)
	
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	# Preparing tmp directory
	if not os.path.exists(TMP_DIR):
		os.mkdir(TMP_DIR)
	# Listening to raw images
	listener()