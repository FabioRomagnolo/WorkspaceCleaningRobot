#!/usr/bin/env python
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

import rospy
import tf
from geometry_msgs.msg import PointStamped


def image_file_to_tensor(path_to_file, precision=torch.float32, device='cpu'):
    # Read the image
    image = cv.imread(path_to_file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Define a transform to convert the image to tensor
    tensor = to_tensor(image).unsqueeze(0)
    # Convert the image to PyTorch tensor according to requirements
    return tensor.to(precision).to(device)


def image2array(image, image_type='rgb'):
    """
    Method to convert image message to Numpy RGB array or RGBA
    """
    image_data = list(image.data)
    image_array = np.array(image_data).astype('uint8')

    if image_type == 'rgba':
        channels = 4
    else:
        channels = 3
    
    reshaped_image = image_array.reshape(image.height, image.width, channels)
    
    if image_type == 'rgba':
        output = cv.cvtColor(reshaped_image, cv.COLOR_BGRA2RGBA)
    else:
        output = cv.cvtColor(reshaped_image, cv.COLOR_BGR2RGB)
    
    return output


def transform_pixels2camera(pixels, camera_info, depth):
    """
    #############################################################
    # Method to transform 2D image pixels into 3D w.r.t. camera #
    #############################################################
    
    @param pixels: Numpy array of 2D pixels of shape (n, 2)
    @param camera_info: Object of the class CameraInfo containing the camera parameters.
                        Ros documentation: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    @param depth: Distance of camera from pixels in meters (e.g. a table)
    """
    # Camera params
    K = np.array(camera_info.K, dtype=np.float32).reshape((3, 3))			# Intrinsic camera matrix
    R = np.array(camera_info.R, dtype=np.float32).reshape((3, 3))		    # Rotation matrix
    P = np.array(camera_info.P, dtype=np.float32).reshape((3, 4))			# Projection camera matrix
    f = (K[0][0], K[1][1])  		                                        # Focal lenght [fx, fy]
    c = (K[0][2], K[1][2])                                                  # Principal point [cx, cy]
    t = np.array([P[0][3], P[1][3], P[2][3]], dtype=np.float32)	    		# Translation [tx, ty, tz]

    # Homogeneous pixel coordinates
    homo_pixels = np.append(pixels, np.ones([pixels.shape[0], 1]), axis=1).astype(np.float32)
    output = []
    for p in homo_pixels:
        
        """
        ### Approach inspired by: https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p 
        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)
        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0], dtype=np.float32)
        cam_world = t + R @ cam
        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # Point scaled along this ray
        p3D = cam_world + distance * unit_vector
        output.append(p3D.tolist())
        """

        ### Simpler and more accurate approach, knowing the exact depth
        # Inspired by https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0
        X = (p[0] - c[0])/f[0] * depth
        Y = (p[1] - c[1])/f[1] * depth
        p3D = [X, Y, depth]
        output.append(p3D)

    output = np.array(output, dtype=np.float32)

    return output


def transform_points(points, ref_frame, init_frame):
    # WARNING! rospy.init_node() must be called before this function!

    tf_listener = tf.TransformListener()
    tf_camera2world = tf_listener.waitForTransform('world', 'camera_link', rospy.Time(0), rospy.Duration(10))

    transformed = []
    for p in points:
        point = PointStamped()
        point.header.frame_id = 'camera_link'
        point.header.stamp = rospy.Time()

        point.point.x = p[0]
        point.point.y = p[1]
        point.point.z = p[2]

        tp = tf_listener.transformPoint('world', point).point

        transformed.append([tp.x, tp.y, tp.z])

    output = np.array(transformed, dtype=np.float32)
    return output
