
set(CMAKE_SOURCE_DIR ..)
set(LINT_COMMAND ${CMAKE_SOURCE_DIR}/scripts/cpp_lint.py)
set(SRC_FILE_EXTENSIONS h hpp hu c cpp cu cc)
set(EXCLUDE_FILE_EXTENSTIONS pb.h pb.cc)
set(LINT_DIRS include src/caffe examples tools python matlab)

cmake_policy(SET CMP0009 NEW)  # supress cmake warning

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
    endforeach()
endforeach()

# find all files that should be excluded
foreach(ext ${EXCLUDE_FILE_EXTENSTIONS})
   