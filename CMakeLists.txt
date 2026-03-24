cmake_minimum_required(VERSION 3.16)

project(ClosestPair LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(closest_pair CP.cpp)

# Apply explicit high optimization only for Release builds.
if(MSVC)
	target_compile_options(closest_pair PRIVATE $<$<CONFIG:Release>:/O2>)
else()
	target_compile_options(closest_pair PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

find_package(Threads REQUIRED)
target_link_libraries(closest_pair PRIVATE Threads::Threads)
