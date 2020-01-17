################################################################################################
# Caffe status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   caffe_status(<text>)
#   caffe_status(<heading> <value1> [<value2> ...])
#   caffe_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(caffe_status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    endif()
  endforeach()

  if(DEFINED status_cond)
    set(status_placeholder_length 23)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif(DEFINED status_then OR DEFINED status_else)
      message(STATUS "${text}")
      set(status_text "${status_placeholder}")
    else()
      set(status_text "${text}")
    endif()

    if(DEFINED status_then OR DEFINED status_else)
      if(${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        message(STATUS "${status_text} ${status_then}")
      else()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        message(STATUS "${status_text} ${status_else}")
      endif()
    else()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      message(STATUS "${status_text} ${status_cond}")
    endif()
  else()
    message(STATUS "${text}")
  endif()
endfunction()


################################################################################################
# Function for fetching Caffe version from git and headers
# Usage:
#   caffe_extract_caffe_version()
function(caffe_extract_caffe_version)
  set(Caffe_GIT_VERSION "unknown")
  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    OUTPUT_VARIABLE Caffe_GIT_VERSION
                    RESULT_VARIABLE __git_result)
    if(NOT ${__git_result} EQUAL 0)
      set(Caffe_GIT_VERSION "unknown")
    endif()
  endif()

  set(Caffe_GIT_VERSION ${Caffe_GIT_VERSION} PARENT_SCOPE)
  set(Caffe_VERSION "<TODO> (Caffe doesn't declare its version in headers)" PARENT_SCOPE)

  # caffe_parse_header(${Caffe_INCLUDE_DIR}/caffe/version.hpp Caffe_VERSION_LINES CAFFE_MAJOR CAFFE_MINOR CAFFE_PATCH)
  # set(Caffe_VERSION "${CAFFE_MAJOR}.${CAFFE_MINOR}.${CAFFE_PATCH}" PARENT_SCOPE)

  # or for #define Caffe_VERSION "x.x.x"
  # caffe_parse_header_single_define(Caffe ${Caffe_INCLUDE_DIR}/caffe/version.hpp Caffe_VERSION)
  # set(Caffe_VERSION ${Caffe_VERSION_STRING} PARENT_SCOPE)

endfunction()


################################################################################################
# Prints accumulatd caffe configuration summary
# Usage:
#   caffe_print_configuration_summary()

function(caffe_print_configuration_summary)
  caffe_extract_caffe_version()
  set(Caffe_VERSION ${Caffe_VERSION} PARENT_SCOPE)

  caffe_merge_flag_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_F