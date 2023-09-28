# Installation

Please clone the ompl repository and add the ITIRRT files to the planners folder:

```
cp ITIRRT.cpp ompl/src/ompl/geometric/planners/rrt/src/
cp ITIRRT.h ompl/src/ompl/geometric/planners/rrt/
```

Then, download the torch library and update the CMakeLists.txt to use the library. 
In ompl/CMakeLists.txt please add the following lines with the correct path.

```
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch/share/cmake/Torch")
find_package(Torch)
include_directories("${TORCH_INCLUDE_DIRS}")
```

Finally, in `ompl/src/ompl/CMakeLists.txt` please add `${TORCH_LIBRARIES}` in the target_link_libraries.
    
# Example

We provide a minimal example on a very simple planning problem to demonstrate how to use our planner.

![example](https://github.com/thibaultbarbie/itirrt/assets/8063351/c26ffd75-9b97-4a17-8a8b-d773ed460138)

The planning problem is to go from the start state to the goal state through a narrow tunnel. 

Please execute the `training.py` to train an MLP model that outputs a path depending on the high of the tunnel. Then, please place the `UsingITIRRT.cpp" file in the demos folder of ompl and add the following to the CMakeLists.txt:

```
add_ompl_demo(demo_UsingITIRRT UsingITIRRT.cpp)
```

After building the ompl repository the `demo_UsingITIRRT` should appear in the bin folder. Please add the tunnel_model.pt which was created by the python script in the bin folder and execute `demo_UsingITIRRT.cpp`. 

If everything was right you should have solved the problem with the ITIRRT planner leveraging your trained model!
