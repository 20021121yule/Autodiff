cmake_minimum_required(VERSION 3.29)
project(autodiff)

set(CMAKE_CXX_STANDARD 20)

# 指定 Eigen 库的路径
# 这里的路径是你的 Eigen 库的 include 目录路径，根据实际情况修改
include_directories(/Users/yule/Desktop/CLion配置/C++库文件/Eigen)

# 这里-O3 -march=native指令可以大幅度提升Eigen库的性能
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 设置输出可执行文件
add_executable(autodiff main.cpp
        Value.cpp
        Value.h
        Value_Scalar.cpp
        Value_Scalar.h
        Autodiff.cpp
        Autodiff.h
)

# 如果 Eigen 需要链接库文件，可以在这里添加链接指令
# target_link_libraries(Eigen <其他库文件>)
