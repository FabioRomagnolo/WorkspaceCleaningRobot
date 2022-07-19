#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2 as cv
import torch
import torchvision
import numpy as np
import rospkg
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import time
import sys
from cv_bridge import CvBridge

class HandsDetectionNode:
    def __init__(self):
        self.package_name = 'my_package'
        self.r = rospkg.RosPack()
        self.saves_dir = os.path.join(self.r.get_path(self.package_name), 'pytorch_saves')
        #self.saves_dir = '/home/user/WorkspaceCleaningRobot/ros_1/catkin_ws/src/my_package/pytorch_saves'
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            num_classes=2)
        self.model.load_state_dict(torch.load(os.path.join(
            self.saves_dir, 'fasterrcnn_model_state_dict'), map_location=torch.device('cpu')))
        self.model.eval()

        # threshold for the scores of the bounding boxes
        self.score_threshold = 0.95

        self.wait_time = 4
        self.old_time = time.time()
        self.bridge = CvBridge()


    def get_final_boxes(out, score_threshold):
        final_boxes = None
        boxes = out[0]['boxes']
        scores = out[0]['scores']
        for box, score in zip(boxes, scores):
            if score > score_threshold:
                if final_boxes is None:
                    final_boxes = torch.clone(box).reshape(1, 4)
                else:
                    final_boxes = torch.cat((final_boxes, box.reshape(1, 4)))
        return final_boxes


    def image2array(self, image):
        """
        Method to convert image message to Numpy RGB array.
        """
        image_data = list(image.data)
        image_array = np.array(image_data).astype('uint8')
        bgr_image = image_array.reshape(image.height, image.width, 3)
        rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        return rgb_image

    def image_callback(self, image_message):
        rospy.loginfo(rospy.get_caller_id() + " Image received!")
        #print(sys.getsizeof(data))

        # Converting image message to Numpy RGB array
        #rgb_array = self.image2array(data)
        cv_image_bgr = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        cv_image_rgb = cv.cvtColor(cv_image_bgr, cv.COLOR_BGR2RGB)
        #plt.imshow(cv_image_rgb)
        #plt.show()
        rgb_tensor = transforms.ToTensor()(cv_image_rgb)

        out = self.model(rgb_tensor.unsqueeze(0))
        print(out)

        if len(out[0]['scores']) == 0:
            if (time.time() - self.old_time) > self.wait_time:
                print(f'No hands detected for {self.wait_time} seconds. Calling node to clean the table...')
                # ...
                self.old_time = time.time()
            else:
                print(f'No hands detected for {int(time.time() - self.old_time)} seconds. Waiting...')
        else:
            print('At least one hand has been detected! Waiting...')
            self.old_time = time.time()

        # im = rgb_array.copy()
        # for box in out[0]['boxes']:
        #     box = box.detach().numpy().astype(int)
        #     cv.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # plt.imshow(im)
        # plt.show()

        # plt.imshow(rgb_array)
        # plt.show()


if __name__ == '__main__':
    hands_detection_node = HandsDetectionNode()
    rospy.init_node('test_hand_detection', anonymous=True)
    rospy.Subscriber("/image_raw", Image, hands_detection_node.image_callback, queue_size=1, buff_size=96)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
