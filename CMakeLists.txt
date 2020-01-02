##############################
PROJECT( vanishingPoint )
##############################
cmake_minimum_required (VERSION 2.6)

# Packages
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

# Includes
include_directories(	
	${OPENCV_INCLUDE_DIR}
	${PROJECT_BINARY_DIR}
	${EIGEN3_INCLUDE_DIR}
)

set(SOURCE_FILES 
	main.cpp
	MSAC.cpp
	MSAC.h
	#lmmin.h
	#lmmin.c
	lsd.h
	lsd.cpp	
)

# ADD_LIBRARY(lsd lsd.cpp)
# Add executable and target link libraries
ADD_EXECUTABLE( vanishingPoint ${SOURCE_FILES})
TARGET_LINK_LIBRARIES( vanishingPoint ${OpenCV_LIBS} ${Eigen_LIBS})


