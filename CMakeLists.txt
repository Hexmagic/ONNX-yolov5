cmake_minimum_required(VERSION 3.0 )
project(app)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package( OpenCV REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
add_executable(main main.cpp  src/detector.cpp)
target_link_libraries(main  ${OpenCV_LIBS})



