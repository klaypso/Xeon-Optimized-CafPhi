# - Find LAPACK library
# This module finds an installed fortran library that implements the LAPACK
# linear-algebra interface (see http://www.netlib.org/lapack/).
#
# The approach follows that taken for the autoconf macro file, acx_lapack.m4
# (distributed at http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).
#
# This module sets the following variables:
#  LAPACK_FOUND - set to true if a library implementing the LAPACK interface is found
#  LAPACK_LIBRARIES - list of libraries (using full path name) for LAPACK

# Note: I do not think it is a good idea to mixup different BLAS/LAPACK versions
# Hence, this script wants to find a Lapack library matching your Blas library

# Do nothing if LAPACK was found before
IF(NOT LAPACK_FOUND)

SET(LAPACK_LIBRARIES)
SET(LAPACK_INFO)

IF(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  FIND_PACKAGE(BLAS)
ELSE(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  FIND_PACKAGE(BLAS REQUIRED)
ENDIF(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)

# Old search lapack script
include(CheckFortranFunctionExists)

macro(Check_Lapack_Libraries LIBRARIES _prefix _name _flags _list _blas)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if (WIN32)
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library} PATHS ENV LIB PATHS ENV PATH)
      else (WIN32)
        if(APPLE)
          find_library(${_prefix}_${_library}_LIBRARY
            NAMES ${_library}
            PATHS /usr/loca