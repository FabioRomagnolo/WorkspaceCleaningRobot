# WorkspaceCleaningRobot
This is an universitary project developed for the "Smart Robotics" course provided by the University of Modena and Reggio Emilia.
The ROS simulation aims to show the capability of robot to accomplish the task of cleaning a workspace in a collaborative context, using techniques of Deep Learning and Computer Vision together with classical Robotics approaches. 

The robot model seen in the simulation is the following real one: [Franka Emika](https://www.franka.de/).

# Ros 1 Noetic implementation
Before starting, install [Ros 1 Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on [Ubuntu 20.04.4 (Focal Fossa)](https://releases.ubuntu.com/20.04/).<br>
[VMware Workstation 16 Player](https://www.vmware.com/it/products/workstation-player/workstation-player-evaluation.html) is recommended to use, but it is not mandatory.

## Requirements
After the Ros 1 installation, clone the repository and update the Ros dependancies:
```
git clone https://github.com/FabioRomagnolo/WorkspaceCleaningRobot.git
cd WorskspaceCleaningRobot/ros_1/catkin_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic
```

Install the control requirements
```
sudo apt-get update
sudo apt-get install ros-noetic-ros-controllers ros-noetic-rqt-joint-trajectory-controller ros-noetic-moveit
```

Install Python 3.8.10 and requirements from relative file
```
sudo apt install python3.8
cd ros_1/catkin_ws/src/my_package/src
pip install -r requirements.txt
```

## How to use
### Build
Build the Ros environment
```
cd ros_1/catkin_ws/
catkin build my_package
```

### Simulate
Execute the simulation following these steps after the build commands:

1. Launch the simulation and wait for Gazebo and Rviz to complete loading:
```
cd ros_1/catkin_ws/
source devel/setup.bash
roslaunch my_package panda_gazebo_moveit.launch
```

2. Open a new terminal inside root folder and type the following command to start the **control** node drom "scripts" folder.<br>
   To understand how it works, take a look at [ros_control](http://wiki.ros.org/ros_control) package and [MoveIt](https://moveit.ros.org/).
```
cd ros_1/catkin_ws/src/my_package/scripts
source ../../../devel/setup.bash
python3 robot_movement.py
```

3. Open a new tab and start the **hands detector** node from "scritps" folder:
```
source ../../../devel/setup.bash
python3 test_hands_detection.py
```

4. Open another tab and start the **dirt detector** node from "src" folder:
```
source ../../../devel/setup.bash
cd ../src
python3 test_dirty_detection_pubsub.py
```

5. Open another tab and start the **camera** node from "src" folder:
```
source ../../../devel/setup.bash
python3 test_camera_pubsub.py
```

6. Open another tab and start the **dirt manager** node from "scripts" folder:
```
source ../../../devel/setup.bash
cd ../scripts
python3 dirt_manager.py
```

7. Open the last tab inside "scripts" folder and **spawn/despawn** person and dirt to see the simulated cleaning process!
```
source ../../../devel/setup.bash
./spawn_person.sh
./spawn_dirt.sh
./despawn_dirt.sh
```

# Ros 2 implementation
***WARNING**: This implementation is incomplete. See Ros 1 implementation above to see execute the full simulation.*

Before starting, install [Ros 2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html) on [Ubuntu 20.04.4 (Focal Fossa)](https://releases.ubuntu.com/20.04/).<br>
[VMware Workstation 16 Player](https://www.vmware.com/it/products/workstation-player/workstation-player-evaluation.html) is recommended to use, but it is not mandatory.

After the Ros 2 installation, install the following requirements for Foxy:
```
sudo apt update
sudo apt install python3-colcon-common-extensions ros-foxy-gazebo-ros-pkgs ros-foxy-joint-state-publisher-gui ros-foxy-xacro
```

Finally, make sure to source the setup file each time you need to work with Ros or simply add the following command to _~/.bashrc_ file to automatically execute it when opening a new shell:
```
source /opt/ros/foxy/setup.bash
```

## How to use
Firstly, ensure that all required dependencies are installed. In the root folder, type:
```
rosdep install -i --from-path src --rosdistro <your_ros_distro> -y
```

Build the package. In the root folder, type the command:
```
colcon build
```

Now, source the setup file:
```
source install/setup.bash
```

Now you can run the simulation in three ways:
- without gazebo:
```
ros2 launch my_package display.launch.py
```
- with gazebo:
```
ros2 launch my_package display_sim.launch.py
```

## Control simulation
Before starting, install the needed packages for Foxy:
```
sudo apt install ros-foxy-ros2-control ros-foxy-ros2-controllers ros-foxy-gazebo-ros2-control
```
Launch the simulation with gazebo:
```
ros2 launch my_package display_sim_control.launch.py
```
Open a new terminal in the same folder and launch the following commands to verify if everything is okay:
```
ros2 control list_hardware_interfaces
ros2 control list_controllers
```
You should see _claimed_ command interfaces and _active_ controllers.

# Background Matting
In order to use [Background Matting](https://grail.cs.washington.edu/projects/background-matting-v2/#/) network you need to get the pretrained weights _.pth_ files
and place them into the _src/my_package/my_package/BackgroundMatting/trained_models_ folder.<br>
You can download the ready-to-use models [here](https://drive.google.com/drive/folders/1vaTjLTk2CoNzMOgeO70Tjsn5DlFF_cJH?usp=sharing).
