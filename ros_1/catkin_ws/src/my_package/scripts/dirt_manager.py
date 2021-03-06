#!/usr/bin/env python
import rospy
from std_msgs.msg import Empty
import rospkg
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Quaternion, Point
import xacro
import random
import tf

'''
$ rossrv info gazebo_msgs/SpawnModel
string model_name
string model_xml
string robot_namespace
geometry_msgs/Pose initial_pose
  geometry_msgs/Point position
    float64 x
    float64 y
    float64 z
  geometry_msgs/Quaternion orientation
    float64 x
    float64 y
    float64 z
    float64 w
string reference_frame
---
bool success
string status_message
'''


# rostopic pub /dirt_manager/spawn_dirt std_msgs/Empty --once

class DirtManager:
    def __init__(self):
        self.package_name = 'my_package'
        self.r = rospkg.RosPack()
        #package_path = self.r.get_path(self.package_name)
        #self.xacro_file = os.path.join(package_path, 'urdf', 'dirt_box.xacro')
        # self.xacro_file = '/home/user/WorkspaceCleaningRobot/ros_1/catkin_ws/src/my_package/urdf/dirt_box.xacro'
        self.xacro_file = os.path.join('..', 'urdf', 'dirt_box.xacro')
        self.num_dirt = 3
        self.current_box = 0
        self.current_boxes = []

    def dirt_despawn_callback(self, data):
        print('Despawning dirt...')
        for dirt_box in self.current_boxes:
            rospy.wait_for_service('/gazebo/delete_model')
            try:
                print(f'Deleting {dirt_box}...')
                delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                resp = delete_model(dirt_box)
                print('Success: ', resp.success)
                print('Status message: ', resp.status_message)
            except Exception as e:
                print('Error in despawning dirt')
                print(e)
        print('Done')
        self.current_boxes = []


    def dirt_spawn_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "Message received! Preparing to spawn dirt...")
        for i in range(self.num_dirt):
            rospy.wait_for_service('/gazebo/spawn_urdf_model')
            try:
                spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
                model_name = f'test_dirt_{self.current_box}'
                #box_x = random.uniform(0.15, 0.85)
                #box_y = random.uniform(-0.65, 0.65)
                box_x = random.uniform(0.3, 0.7)
                box_y = random.uniform(-0.5, 0.5)
                box_z = 0.45
                mappings = {
                    'dirt_box_name': f'dirt_box_{self.current_box}',
                    'dirt_box_size_xyz': '0.07 0.07 0.07',
                    'dirt_box_xyz': f'{box_x} {box_y} {box_z}'
                }
                self.current_boxes.append(model_name)
                self.current_box += 1
                model_xml = xacro.process_file(self.xacro_file, mappings=mappings).toxml()

                initial_pose = Pose(
                    Point(0.0, 0.0, 0.0),
                    Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0))
                )

                resp1 = spawn_model(
                    model_name=model_name,
                    model_xml=model_xml,
                    #initial_pose=initial_pose,
                )
                print('Success: ', resp1.success)
                print('Status message: ', resp1.status_message)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        return (resp1.success, resp1.status_message)
            


if __name__ == '__main__':
    dirt_manager = DirtManager()
    rospy.init_node('test_dirt_manager', anonymous=True)
    rospy.Subscriber("/dirt_manager/spawn_dirt", Empty, dirt_manager.dirt_spawn_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.Subscriber("/dirt_manager/despawn_dirt", Empty, dirt_manager.dirt_despawn_callback)
    rospy.spin()
