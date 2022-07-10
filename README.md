# WorkspaceCleaningRobot

Possibile physical robot: YuMi IRB 14050

# Ros 1 implementation
Before starting, install [Ros 1 Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on [Ubuntu 20.04.4 (Focal Fossa)](https://releases.ubuntu.com/20.04/).<br>
[VMware Workstation 16 Player](https://www.vmware.com/it/products/workstation-player/workstation-player-evaluation.html) is recommended to use, but it is not mandatory.

After the Ros 1 installation, update the dependancies:
```
cd ros_1/catkin_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic
```

Install the controllers
```
sudo apt-get update
sudo apt-get install ros-noetic-ros-controllers
```

## How to use
Build the project
```
cd ros_1/catkin_ws/
catkin build my_package
source devel/setup.bash
```

Launch the robot
```
roslaunch my_package gazebo.launch
```

# Ros 2 implementation
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
