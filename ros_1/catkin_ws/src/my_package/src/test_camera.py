#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def callback(image):
	#print(image.data)
	#print(image.width) # 640
	#print(image.height) # 480
	#print(len(image.data)) # 921600 = 640*480*3
	#print(image.encoding) # bgr8
	image_data = list(image.data)
	image_array = np.array(image_data).astype('uint8')
	bgr_image = image_array.reshape(480, 640, 3)
	rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
	plt.imshow(rgb_image)
	plt.show()
	rospy.loginfo(rospy.get_caller_id() + "I heard the image")
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
	listener()