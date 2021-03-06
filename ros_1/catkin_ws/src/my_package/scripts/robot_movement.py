#!/usr/bin/env python

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import UInt32MultiArray, Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np
import tf
import std_msgs.msg

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

        self.cleaning_state_pub = rospy.Publisher('cleaning_state', std_msgs.msg.String, queue_size=10)
        self.dirt_manager_pub = rospy.Publisher('dirt_manager/despawn_dirt', std_msgs.msg.Empty, queue_size=10)
    
    def return_to_home_conf(self):
        joint_goal = obj.move_group.get_current_joint_values()
        joint_goal[0] = 0.0
        joint_goal[1] = 0.0705
        joint_goal[2] = 0.0
        joint_goal[3] = -0.3532
        joint_goal[4] = 0.0
        joint_goal[5] = 0.4349
        joint_goal[6] = 0.0

        obj.move_group.go(joint_goal, wait=True)
        obj.move_group.stop()
        self.move_group.set_start_state(self.move_group.get_current_state())

      
    def coordinates_cb(self, msg):
        print('------ DIRTY COORDINATES RECEIVED ------')
        
        # Publishing state
        self.cleaning_state_pub.publish('cleaning')

        points = np.array(msg.data).reshape(-1, 3)

        waypoints = []
        waypoints.append(self.move_group.get_current_pose().pose)
        for i, p in enumerate(points):
            print(f'{i}. Current point: {p}')
            # Moving robot end effector to the 3D point
            pose_goal = Pose()
            pose_goal.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0.0, np.pi, 0.0))

            pose_goal.position.x = p[0]
            pose_goal.position.y = p[1]
            #pose_goal.position.z = p[2]+0.23
            pose_goal.position.z = 0.6

            waypoints.append(pose_goal)

        '''
        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        # eef_step = 0.01
        eef_step = 0.01
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, eef_step, 0.0  # waypoints to follow  # eef_step  # jump_threshold
        )  

        print('Fraction: ', fraction)
        # print('Trajectory: ', plan)

        print("- Executing task ...")
        self.move_group.execute(plan, wait=True)

        # Returning to home configuration of the joints
        print("- Returning to home configuration ...")
        self.return_to_home_conf()
        '''
        
        pose_targets = waypoints
        for i, goal in enumerate(pose_targets):
            if i == 0:
                continue
            if i == 1:
                print(f"- Planning moving to pose {i} from pose {i-1}...")


                self.move_group.set_pose_target(goal)
                plan = self.move_group.go(wait=True)
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                self.move_group.set_start_state(self.move_group.get_current_state())

                # plan, fraction = self.move_group.compute_cartesian_path(
                #     [waypoints[i]], 0.01, 0.0  # waypoints to follow  # eef_step  # jump_threshold
                # )
                # print('Fraction: ', fraction)
                # self.move_group.execute(plan, wait=True)
                # self.move_group.stop()
                # self.move_group.set_start_state(self.move_group.get_current_state())

                continue

            # move_points = [
            #     self.move_group.get_current_pose().pose,
            #     waypoints[i]
            # ]
            print(f"- Planning moving to pose {i} from pose {i-1}...")

            plan, fraction = self.move_group.compute_cartesian_path(
                [waypoints[i]], 0.01, 0.0  # waypoints to follow  # eef_step  # jump_threshold
            )
            print('Fraction: ', fraction)

            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.set_start_state(self.move_group.get_current_state())
            
            '''
            print(f"- Planning moving to pose {i} from pose {i-1}...")
            self.move_group.set_pose_target(goal)
            ## Now, we call the planner to compute the plan and execute it.
            print("- Executing task ...")
            plan = self.move_group.go(wait=True)
            # Calling `stop()` ensures that there is no residual movement
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.move_group.set_start_state(self.move_group.get_current_state())
            '''

        print('Finished. Despawning dirt...')
        self.dirt_manager_pub.publish()

        print("- Finished. Returning to home configuration ...")
        self.return_to_home_conf()

        # Publishing state
        self.cleaning_state_pub.publish('finished cleaning')



        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        #self.move_group.clear_pose_targets()
        

        '''
        self.move_group.set_pose_targets(pose_targets)

        print("- Executing task ...")
        ## Now, we call the planner to compute the plan and execute it.
        plan = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        #cur_state = self.move_group.get_current_state()
        #print('Current state: ', cur_state)
        #self.move_group.set_start_state(cur_state)

        #self.move_group.set_start_state_to_current_state()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        #current_pose = self.move_group.get_current_pose().pose
        #return all_close(pose_goal, current_pose, 0.01)
        '''
        print("---------------------------------------")
        return


if __name__=='__main__':
    obj = MoveToClean()
    obj.return_to_home_conf()
    rospy.spin()