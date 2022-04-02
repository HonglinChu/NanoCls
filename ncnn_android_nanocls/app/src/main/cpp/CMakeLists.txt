# cmake version
cmake_minimum_required(VERSION 3.18.1)

# 1 include 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/
        ${CMAKE_SOURCE_DIR}/include/ncnn
)

# 2 将cpp目录下， 自己写的cpp文件编译成一个 so 或者 a 库，指定自己编写的cpp文件所在目录
# 设置源文件的目录, 指定 .CPP 的文件路径
file(GLOB NANOCLS_SRC ${CMAKE_SOURCE_DIR}/*.h
        ${CMAKE_SOURCE_DIR}/*.cpp)

# 设置一个变量
set(NANOCLS_COMPILE_CODE ${NANOCLS_SRC})

# 3 导入第三方库 ncnn
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/lib/ncnn/${ANDROID_ABI}/libncnn.a)
#find_package(ncnn)，它会自动设定正确的链接库及其顺序

# 导入第三方库vulkan，来提供GPU的支持
#add_library(glslang STATIC IMPORTED)
#add_library(OGLCompiler STATIC IMPORTED)
#add_library(OSDependent STATIC IMPORTED)
#add_library(SPIRV STATIC IMPORTED)
#set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/ncnnvulkan/${ANDROID_ABI}/libglslang.a)
#set_target_properties(OGLCompiler PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/ncnnvulkan/${ANDROID_ABI}/libOGLCompiler.a)
#set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/ncnnvulkan/${ANDROID_ABI}/libOSDependent.a)
#set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/ncnnvulkan/${ANDROID_ABI}/libSPIRV.a)

# openmp 是一种用于共享内存并行系统的多线程程序设计方案
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")
if(DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

# 禁用 rtti 和 exceptions C++的exception和RTTI功能在NDK中默认是关闭的，
# 但是可以通过配置打开的： set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")
# Exception和RTTI会显著的增加包的体积，所以非必须的时候，没有必要使用
# 通过RTTI，能够通过基类的指针或引用来检索其所指对象的实际类型，即运行时获取对象的实际类型。C++通过下面两个操作符提供RTTI。
# 对于开发者来说，exception可以帮助快速定位问题，而对于使用者并不是那么重要，这里可以去掉。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

# 4 将代码编译为动态链接库
add_library(nanocls SHARED ${NANOCLS_COMPILE_CODE})
target_link_libraries(nanocls ncnn android z log jnigraphics)
