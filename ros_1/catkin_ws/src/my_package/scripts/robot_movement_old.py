#!/usr/bin/env python

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import UInt32MultiArray, Float32MultiArray
import numpy as np
import tf

# creating a class
class MoveToClean:
    def __init__(self):
        # initialise moveit commander nad rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_to_clean", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        #self.pose_subscriber = rospy.Subscriber('/cleaning_coordinates',Pose, self.coordinates_cb)
        self.pose_subscriber = rospy.Subscriber('/dirty2world', Float32MultiArray, self.coordinates_cb)

        self.cleaning_pose = []
      
    def coordinates_cb(self, msg):
        print('Got Coordinates')

        points = np.array(msg.data).reshape(-1, 3)
        
        for p in points:
            print('Current point: ', p)
            # Moving robot end effector to the 3D point
            pose_goal = Pose()
            pose_goal.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0.0, np.pi, 0.0))
            pose_goal.position.x = p[0]
            pose_goal.position.y = p[1]
            pose_goal.position.z = p[2]+0.3

            self.move_group.set_pose_target(pose_goal)

            ## Now, we call the planner to compute the plan and execute it.
            plan = self.move_group.go(wait=True)
            # Calling `stop()` ensures that there is no residual movement
            self.move_group.stop()
            # It is always good to clear your targets after planning with poses.
            # Note: there is no equivalent function for clear_joint_value_targets()
            self.move_group.clear_pose_targets()

            cur_state = self.move_group.get_current_state()
            print('Current state: ', cur_state)
            self.move_group.set_start_state(cur_state)

            #self.move_group.set_start_state_to_current_state()

            ## END_SUB_TUTORIAL

            # For testing:
            # Note that since this section of code will not be included in the tutorials
            # we use the class variable rather than the copied state variable
            #current_pose = self.move_group.get_current_pose().pose
            #return all_close(pose_goal, current_pose, 0.01)
        return


if __name__=='__main__':
    obj = MoveToClean()
    rospy.spin()