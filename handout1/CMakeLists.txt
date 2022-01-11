cmake_minimum_required(VERSION 3.5.1)

project (exercise)

# Libraries
# find_package(OpenCV REQUIRED PATHS ${CMAKE_CURRENT_LIST_DIR}/lib/opencv)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(cvt src/given.hpp src/given.cpp)
set_target_properties(cvt PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_link_libraries(cvt PUBLIC ${OpenCV_LIBS})

add_library(cvs src/yours.hpp src/yours.cpp)
set_target_properties(cvs PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_link_libraries(cvs PUBLIC ${OpenCV_LIBS} cvt)

# Application
add_executable(app src/main.cpp)
target_link_libraries(app ${OpenCV_LIBS} cvt cvs)

# Testing
add_library(catch_main test/catch_main.cpp)
set_target_properties(catch_main PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
add_executable(test_app test/tests.cpp)
target_link_libraries(test_app catch_main cvt cvs)
