# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.22.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.22.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build

# Include any dependencies generated for this target.
include CMakeFiles/nanocls_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nanocls_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nanocls_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nanocls_demo.dir/flags.make

CMakeFiles/nanocls_demo.dir/main.cpp.o: CMakeFiles/nanocls_demo.dir/flags.make
CMakeFiles/nanocls_demo.dir/main.cpp.o: ../main.cpp
CMakeFiles/nanocls_demo.dir/main.cpp.o: CMakeFiles/nanocls_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nanocls_demo.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nanocls_demo.dir/main.cpp.o -MF CMakeFiles/nanocls_demo.dir/main.cpp.o.d -o CMakeFiles/nanocls_demo.dir/main.cpp.o -c /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/main.cpp

CMakeFiles/nanocls_demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nanocls_demo.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/main.cpp > CMakeFiles/nanocls_demo.dir/main.cpp.i

CMakeFiles/nanocls_demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nanocls_demo.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/main.cpp -o CMakeFiles/nanocls_demo.dir/main.cpp.s

# Object files for target nanocls_demo
nanocls_demo_OBJECTS = \
"CMakeFiles/nanocls_demo.dir/main.cpp.o"

# External object files for target nanocls_demo
nanocls_demo_EXTERNAL_OBJECTS =

nanocls_demo: CMakeFiles/nanocls_demo.dir/main.cpp.o
nanocls_demo: CMakeFiles/nanocls_demo.dir/build.make
nanocls_demo: ../ncnn/lib/libncnn.a
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_stitching.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_superres.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_videostab.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_aruco.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_bgsegm.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_bioinspired.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ccalib.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dnn_objdetect.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dpm.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_face.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_freetype.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_fuzzy.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_hfs.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_img_hash.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_line_descriptor.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_optflow.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_reg.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_rgbd.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_saliency.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_sfm.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_stereo.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_structured_light.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_surface_matching.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_tracking.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xfeatures2d.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ximgproc.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xobjdetect.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xphoto.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_highgui.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_videoio.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_shape.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_phase_unwrapping.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dnn.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_video.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_datasets.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ml.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_plot.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_imgcodecs.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_objdetect.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_calib3d.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_features2d.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_flann.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_photo.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_imgproc.3.4.15.dylib
nanocls_demo: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_core.3.4.15.dylib
nanocls_demo: CMakeFiles/nanocls_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nanocls_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nanocls_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nanocls_demo.dir/build: nanocls_demo
.PHONY : CMakeFiles/nanocls_demo.dir/build

CMakeFiles/nanocls_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nanocls_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nanocls_demo.dir/clean

CMakeFiles/nanocls_demo.dir/depend:
	cd /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build /Users/chuhonglin/Desktop/NCNN_MNN_Demo/NCNN/MacOS/ncnn-macos-nanocls/build/CMakeFiles/nanocls_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nanocls_demo.dir/depend

