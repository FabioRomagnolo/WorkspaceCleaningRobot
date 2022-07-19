PACKAGE_PATH=$(rospack find my_package)
echo $PACKAGE_PATH
xacro $PACKAGE_PATH/urdf/person_new.xacro | rosrun gazebo_ros spawn_model -urdf -stdin -model person