project( SimpleDBSCAN )
cmake_minimum_required( VERSION 3.5 )


add_library(${PROJECT_NAME} SHARED
	src/kdtree.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

# Install binaries
install(TARGETS ${PROJECT_NAME}
	EXPORT 				${PROJECT_NAME}-targets
	ARCHIVE       DESTINATION lib
	RUNTIME       DESTINATION bin
	LIBRARY       DESTINATION lib
	INCLUDES			DESTINATION include
)

# Export the library interface
install(EXPORT ${PROJECT_NAME}-targets
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION lib/cmake/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)