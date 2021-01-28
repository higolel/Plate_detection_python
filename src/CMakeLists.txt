cmake_minimum_required(VERSION 2.8.3)
project(plate_detection)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
)

#catkin_python_setup()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11")

catkin_package(
	CATKIN_DEPENDS roscpp rospy
)

include_directories(
	${catkin_INCLUDE_DIRS}
)

#add_executable(${PROJECT_NAME}_node scripts/example_video.py)

install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
	PATTERN ".svn" EXCLUDE
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY launch model scripts font
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


