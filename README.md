# Accelerated RRT* with LDV
Implementation of the Accelerated RRT* with Local Directional Visibility (LDV) Algorithm

### Authors

**Vikram Setty (vikrams@umd.edu)**

**Vinay Lanka (vlanka@umd.edu)**

## Algorithm visualised

This Python script (`grid_world/main.py`) demonstrates the Accelerated RRT* with LDV algorithm (based on the paper linked [here](https://arxiv.org/abs/2207.08283)) for pathfinding in a 2D grid environment using PyGame for visualization. The script generates a map with obstacles and walls, allows the user to input a start and goal position, and then visualizes the process of finding the shortest path from the start to the goal using the algorithm.

## Dependencies
The dependencies for this Python 3 project include the following:
<ul>
<li> NumPy
<li> PyGame
<li> ImageIO
</ul>
They can be installed using the following commands.

```sh
    pip3 install numpy
    pip3 install pygame
    pip3 install opencv-python
    pip3 install imageio
    pip3 install imageio-ffmpeg
```

### Running the Script

To run the script, use the following commands:

```sh
    cd grid_world/
    python3 main.py
```
Once the animation window opens, do the following:
<ul><li> Click any area (except obstacles) to mark the start node
<li>Click on another point (except obstacles) to mark the target/goal node
<li>Click on the space key to start the RRT* Simulation.</ul>

Do note that the map type being run on and other parametets specific to the algorithm can be changed inside `grid_world/main.py`.

## ROS and Gazebo Implementation

The second part of the project replicates the algorithm devised in a Gazebo simulation environment using ROS 2. An appropriate conversion to the linear and angular velocity input to the non-holonoimic differential-drive Turtlebot 3 Waffle-Pi robot is given to enable the robot to follow the same trajectory in the same map in Gazebo (scaled by 1mm/pixel).

### Dependencies

This package is tested to run in a ROS 2 Galactic setup on Ubuntu 20.04. Please prefer this environment specification before proceeding. In addition, a couple more dependencies needed to run the package can be installed using the commands below.


### Building the Package

To build the package, execute.
```sh
    # Navigate to the src directory of your ROS 2 workspace
    cd acc_rrt_star_ws/
    # Install all dependencies
    rosdep install --from-paths src -y --ignore-src
    # Build the package and source the setup file
    colcon build && source install/setup.bash
```
To launch the environment, execute the following commands.
```sh
    # Navigate to the src directory of your ROS 2 workspace
    cd acc_rrt_star_ws/
    # Source the workspace
    source install/setup.bash
    # Export TURTLEBOT3_MODEL
    export TURTLEBOT3_MODEL=waffle
    # Run the launch file to start the gazebo environment
    ros2 launch acc_rrt_star competition_world.launch.py
```
To run a sample trajectory, execute the following commands. You will also be prompted to input only the goal node position (in a way similar to part 1). The start position and other parameters are fixed. A sample goal position is (5750,1000) i.e. the same coordinates used in Part 1.
```sh
    # Navigate to the src directory of your ROS 2 workspace
    cd ros2_ws
    # Source the workspace
    source install/setup.bash
    # Run the file to start the sample trajectory
    ros2 run acc_rrt_star main.py
```

### Simulation Video

The YouTube link to a sample simulation video of this project is embedded below. The link to the same can also be found [here](https://youtu.be/BdJfilJLqzQ).

[![Video](https://img.youtube.com/vi/BdJfilJLqzQ/maxresdefault.jpg)](https://youtu.be/BdJfilJLqzQ)
