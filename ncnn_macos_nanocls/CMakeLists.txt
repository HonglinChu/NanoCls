cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)
project(nanocls_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # Debug模式 

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "Current build dir:" ${CMAKE_CURRENT_LIST_DIR})

set(OpenCV_STATIC ON) 
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv@3/3.4.15/share/OpenCV")  
find_package(OpenCV REQUIRED)

# 引入外部库的方式1, 制定外部库的搜索路径 
# link_directories(${ROOT_DIR}/ncnn/lib)  

# 引入外部库的方式2 以导入外部库的方式(不会重新编译)重新构建一个ncnn库, 然后可执行程序链接到该库
add_library(ncnn STATIC IMPORTED) # 导入 ncnn dynamic lib 
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION  ${ROOT_DIR}/ncnn/lib/libncnn.a)

# /opt/homebrew/Cellar/libomp/13.0.0 

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find_package(OpenCV REQUIRED)

# find_package(ncnn REQUIRED)
# if(NOT TARGET ncnn) 
#     message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
# else()
#     message("ncnn FOUND ")
# endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn/ 
)

add_executable(nanocls_demo main.cpp) 

target_link_libraries(
    nanocls_demo 
    ncnn
    ${OpenCV_LIBS}
) 
