
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file defines the public API for Google Test.  It should be
// included by any test program that uses Google Test.
//
// IMPORTANT NOTE: Due to limitation of the C++ language, we have to
// leave some internal implementation details in this header file.
// They are clearly marked by comments like this:
//
//   // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
//
// Such code is NOT meant to be used by a user directly, and is subject
// to CHANGE WITHOUT NOTICE.  Therefore DO NOT DEPEND ON IT in a user
// program!
//
// Acknowledgment: Google Test borrowed the idea of automatic test
// registration from Barthelemy Dagenais' (barthelemy@prologique.com)
// easyUnit framework.

#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_H_

#include <limits>
#include <vector>

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares functions and macros used internally by
// Google Test.  They are subject to change without notice.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan)
//
// Low-level types and utilities for porting Google Test to various
// platforms.  They are subject to change without notice.  DO NOT USE
// THEM IN USER CODE.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

// The user can define the following macros in the build script to
// control Google Test's behavior.  If the user doesn't define a macro
// in this list, Google Test will define it.
//
//   GTEST_HAS_CLONE          - Define it to 1/0 to indicate that clone(2)
//                              is/isn't available.
//   GTEST_HAS_EXCEPTIONS     - Define it to 1/0 to indicate that exceptions
//                              are enabled.
//   GTEST_HAS_GLOBAL_STRING  - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::string, which is different to std::string).
//   GTEST_HAS_GLOBAL_WSTRING - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::wstring, which is different to std::wstring).
//   GTEST_HAS_POSIX_RE       - Define it to 1/0 to indicate that POSIX regular
//                              expressions are/aren't available.
//   GTEST_HAS_PTHREAD        - Define it to 1/0 to indicate that <pthread.h>
//                              is/isn't available.
//   GTEST_HAS_RTTI           - Define it to 1/0 to indicate that RTTI is/isn't
//                              enabled.
//   GTEST_HAS_STD_WSTRING    - Define it to 1/0 to indicate that
//                              std::wstring does/doesn't work (Google Test can
//                              be used where std::wstring is unavailable).
//   GTEST_HAS_TR1_TUPLE      - Define it to 1/0 to indicate tr1::tuple
//                              is/isn't available.
//   GTEST_HAS_SEH            - Define it to 1/0 to indicate whether the
//                              compiler supports Microsoft's "Structured
//                              Exception Handling".
//   GTEST_HAS_STREAM_REDIRECTION
//                            - Define it to 1/0 to indicate whether the
//                              platform supports I/O stream redirection using
//                              dup() and dup2().
//   GTEST_USE_OWN_TR1_TUPLE  - Define it to 1/0 to indicate whether Google
//                              Test's own tr1 tuple implementation should be
//                              used.  Unused when the user sets
//                              GTEST_HAS_TR1_TUPLE to 0.
//   GTEST_LINKED_AS_SHARED_LIBRARY
//                            - Define to 1 when compiling tests that use
//                              Google Test as a shared library (known as
//                              DLL on Windows).
//   GTEST_CREATE_SHARED_LIBRARY
//                            - Define to 1 when compiling Google Test itself
//                              as a shared library.

// This header defines the following utilities:
//
// Macros indicating the current platform (defined to 1 if compiled on
// the given platform; otherwise undefined):
//   GTEST_OS_AIX      - IBM AIX
//   GTEST_OS_CYGWIN   - Cygwin
//   GTEST_OS_HPUX     - HP-UX
//   GTEST_OS_LINUX    - Linux
//     GTEST_OS_LINUX_ANDROID - Google Android
//   GTEST_OS_MAC      - Mac OS X
//   GTEST_OS_NACL     - Google Native Client (NaCl)
//   GTEST_OS_SOLARIS  - Sun Solaris
//   GTEST_OS_SYMBIAN  - Symbian
//   GTEST_OS_WINDOWS  - Windows (Desktop, MinGW, or Mobile)
//     GTEST_OS_WINDOWS_DESKTOP  - Windows Desktop
//     GTEST_OS_WINDOWS_MINGW    - MinGW
//     GTEST_OS_WINDOWS_MOBILE   - Windows Mobile
//   GTEST_OS_ZOS      - z/OS
//
// Among the platforms, Cygwin, Linux, Max OS X, and Windows have the
// most stable support.  Since core members of the Google Test project
// don't have access to other platforms, support for them may be less
// stable.  If you notice any problems on your platform, please notify
// googletestframework@googlegroups.com (patches for fixing them are
// even more welcome!).
//
// Note that it is possible that none of the GTEST_OS_* macros are defined.
//
// Macros indicating available Google Test features (defined to 1 if
// the corresponding feature is supported; otherwise undefined):
//   GTEST_HAS_COMBINE      - the Combine() function (for value-parameterized
//                            tests)
//   GTEST_HAS_DEATH_TEST   - death tests
//   GTEST_HAS_PARAM_TEST   - value-parameterized tests
//   GTEST_HAS_TYPED_TEST   - typed tests
//   GTEST_HAS_TYPED_TEST_P - type-parameterized tests
//   GTEST_USES_POSIX_RE    - enhanced POSIX regex is used. Do not confuse with
//                            GTEST_HAS_POSIX_RE (see above) which users can
//                            define themselves.
//   GTEST_USES_SIMPLE_RE   - our own simple regex is used;
//                            the above two are mutually exclusive.
//   GTEST_CAN_COMPARE_NULL - accepts untyped NULL in EXPECT_EQ().
//
// Macros for basic C++ coding:
//   GTEST_AMBIGUOUS_ELSE_BLOCKER_ - for disabling a gcc warning.
//   GTEST_ATTRIBUTE_UNUSED_  - declares that a class' instances or a
//                              variable don't have to be used.
//   GTEST_DISALLOW_ASSIGN_   - disables operator=.
//   GTEST_DISALLOW_COPY_AND_ASSIGN_ - disables copy ctor and operator=.
//   GTEST_MUST_USE_RESULT_   - declares that a function's result must be used.
//
// Synchronization:
//   Mutex, MutexLock, ThreadLocal, GetThreadCount()
//                  - synchronization primitives.
//   GTEST_IS_THREADSAFE - defined to 1 to indicate that the above
//                         synchronization primitives have real implementations
//                         and Google Test is thread-safe; or 0 otherwise.
//
// Template meta programming:
//   is_pointer     - as in TR1; needed on Symbian and IBM XL C/C++ only.
//   IteratorTraits - partial implementation of std::iterator_traits, which
//                    is not available in libCstd when compiled with Sun C++.
//
// Smart pointers:
//   scoped_ptr     - as in TR2.
//
// Regular expressions:
//   RE             - a simple regular expression class using the POSIX
//                    Extended Regular Expression syntax on UNIX-like
//                    platforms, or a reduced regular exception syntax on
//                    other platforms, including Windows.
//
// Logging:
//   GTEST_LOG_()   - logs messages at the specified severity level.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.
//
// Stdout and stderr capturing:
//   CaptureStdout()     - starts capturing stdout.
//   GetCapturedStdout() - stops capturing stdout and returns the captured
//                         string.
//   CaptureStderr()     - starts capturing stderr.
//   GetCapturedStderr() - stops capturing stderr and returns the captured
//                         string.
//
// Integer types:
//   TypeWithSize   - maps an integer to a int type.
//   Int32, UInt32, Int64, UInt64, TimeInMillis
//                  - integers of known sizes.
//   BiggestInt     - the biggest signed integer type.
//
// Command-line utilities:
//   GTEST_FLAG()       - references a flag.
//   GTEST_DECLARE_*()  - declares a flag.
//   GTEST_DEFINE_*()   - defines a flag.
//   GetArgvs()         - returns the command line as a vector of strings.
//
// Environment variable utilities:
//   GetEnv()             - gets the value of an environment variable.
//   BoolFromGTestEnv()   - parses a bool environment variable.
//   Int32FromGTestEnv()  - parses an Int32 environment variable.
//   StringFromGTestEnv() - parses a string environment variable.

#include <ctype.h>   // for isspace, etc
#include <stddef.h>  // for ptrdiff_t
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef _WIN32_WCE
# include <sys/types.h>
# include <sys/stat.h>
#endif  // !_WIN32_WCE

#include <iostream>  // NOLINT
#include <sstream>  // NOLINT
#include <string>  // NOLINT

#define GTEST_DEV_EMAIL_ "googletestframework@@googlegroups.com"
#define GTEST_FLAG_PREFIX_ "gtest_"
#define GTEST_FLAG_PREFIX_DASH_ "gtest-"
#define GTEST_FLAG_PREFIX_UPPER_ "GTEST_"
#define GTEST_NAME_ "Google Test"
#define GTEST_PROJECT_URL_ "http://code.google.com/p/googletest/"

// Determines the version of gcc that is used to compile this.
#ifdef __GNUC__
// 40302 means version 4.3.2.
# define GTEST_GCC_VER_ \
    (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif  // __GNUC__

// Determines the platform on which Google Test is compiled.
#ifdef __CYGWIN__
# define GTEST_OS_CYGWIN 1
#elif defined __SYMBIAN32__
# define GTEST_OS_SYMBIAN 1
#elif defined _WIN32
# define GTEST_OS_WINDOWS 1
# ifdef _WIN32_WCE
#  define GTEST_OS_WINDOWS_MOBILE 1
# elif defined(__MINGW__) || defined(__MINGW32__)
#  define GTEST_OS_WINDOWS_MINGW 1
# else
#  define GTEST_OS_WINDOWS_DESKTOP 1
# endif  // _WIN32_WCE
#elif defined __APPLE__
# define GTEST_OS_MAC 1
#elif defined __linux__
# define GTEST_OS_LINUX 1
# ifdef ANDROID
#  define GTEST_OS_LINUX_ANDROID 1
# endif  // ANDROID
#elif defined __MVS__
# define GTEST_OS_ZOS 1
#elif defined(__sun) && defined(__SVR4)
# define GTEST_OS_SOLARIS 1
#elif defined(_AIX)
# define GTEST_OS_AIX 1
#elif defined(__hpux)
# define GTEST_OS_HPUX 1
#elif defined __native_client__
# define GTEST_OS_NACL 1
#endif  // __CYGWIN__

// Brings in definitions for functions used in the testing::internal::posix
// namespace (read, write, close, chdir, isatty, stat). We do not currently
// use them on Windows Mobile.
#if !GTEST_OS_WINDOWS
// This assumes that non-Windows OSes provide unistd.h. For OSes where this
// is not the case, we need to include headers that provide the functions
// mentioned above.
# include <unistd.h>
# if !GTEST_OS_NACL
// TODO(vladl@google.com): Remove this condition when Native Client SDK adds
// strings.h (tracked in
// http://code.google.com/p/nativeclient/issues/detail?id=1175).
#  include <strings.h>  // Native Client doesn't provide strings.h.
# endif
#elif !GTEST_OS_WINDOWS_MOBILE
# include <direct.h>
# include <io.h>
#endif

// Defines this to true iff Google Test can use POSIX regular expressions.
#ifndef GTEST_HAS_POSIX_RE
# define GTEST_HAS_POSIX_RE (!GTEST_OS_WINDOWS)
#endif

#if GTEST_HAS_POSIX_RE

// On some platforms, <regex.h> needs someone to define size_t, and
// won't compile otherwise.  We can #include it here as we already
// included <stdlib.h>, which is guaranteed to define size_t through
// <stddef.h>.
# include <regex.h>  // NOLINT

# define GTEST_USES_POSIX_RE 1

#elif GTEST_OS_WINDOWS

// <regex.h> is not available on Windows.  Use our own simple regex
// implementation instead.
# define GTEST_USES_SIMPLE_RE 1

#else

// <regex.h> may not be available on this platform.  Use our own
// simple regex implementation instead.
# define GTEST_USES_SIMPLE_RE 1

#endif  // GTEST_HAS_POSIX_RE

#ifndef GTEST_HAS_EXCEPTIONS
// The user didn't tell us whether exceptions are enabled, so we need
// to figure it out.
# if defined(_MSC_VER) || defined(__BORLANDC__)
// MSVC's and C++Builder's implementations of the STL use the _HAS_EXCEPTIONS
// macro to enable exceptions, so we'll do the same.
// Assumes that exceptions are enabled by default.
#  ifndef _HAS_EXCEPTIONS
#   define _HAS_EXCEPTIONS 1
#  endif  // _HAS_EXCEPTIONS
#  define GTEST_HAS_EXCEPTIONS _HAS_EXCEPTIONS
# elif defined(__GNUC__) && __EXCEPTIONS
// gcc defines __EXCEPTIONS to 1 iff exceptions are enabled.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__SUNPRO_CC)
// Sun Pro CC supports exceptions.  However, there is no compile-time way of
// detecting whether they are enabled or not.  Therefore, we assume that
// they are enabled unless the user tells us otherwise.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__IBMCPP__) && __EXCEPTIONS
// xlC defines __EXCEPTIONS to 1 iff exceptions are enabled.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__HP_aCC)
// Exception handling is in effect by default in HP aCC compiler. It has to
// be turned of by +noeh compiler option if desired.
#  define GTEST_HAS_EXCEPTIONS 1
# else
// For other compilers, we assume exceptions are disabled to be
// conservative.
#  define GTEST_HAS_EXCEPTIONS 0
# endif  // defined(_MSC_VER) || defined(__BORLANDC__)
#endif  // GTEST_HAS_EXCEPTIONS

#if !defined(GTEST_HAS_STD_STRING)
// Even though we don't use this macro any longer, we keep it in case
// some clients still depend on it.
# define GTEST_HAS_STD_STRING 1
#elif !GTEST_HAS_STD_STRING
// The user told us that ::std::string isn't available.
# error "Google Test cannot be used where ::std::string isn't available."
#endif  // !defined(GTEST_HAS_STD_STRING)

#ifndef GTEST_HAS_GLOBAL_STRING
// The user didn't tell us whether ::string is available, so we need
// to figure it out.

# define GTEST_HAS_GLOBAL_STRING 0

#endif  // GTEST_HAS_GLOBAL_STRING

#ifndef GTEST_HAS_STD_WSTRING
// The user didn't tell us whether ::std::wstring is available, so we need
// to figure it out.
// TODO(wan@google.com): uses autoconf to detect whether ::std::wstring
//   is available.

// Cygwin 1.7 and below doesn't support ::std::wstring.
// Solaris' libc++ doesn't support it either.  Android has
// no support for it at least as recent as Froyo (2.2).
# define GTEST_HAS_STD_WSTRING \
    (!(GTEST_OS_LINUX_ANDROID || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS))

#endif  // GTEST_HAS_STD_WSTRING

#ifndef GTEST_HAS_GLOBAL_WSTRING
// The user didn't tell us whether ::wstring is available, so we need
// to figure it out.
# define GTEST_HAS_GLOBAL_WSTRING \
    (GTEST_HAS_STD_WSTRING && GTEST_HAS_GLOBAL_STRING)
#endif  // GTEST_HAS_GLOBAL_WSTRING

// Determines whether RTTI is available.
#ifndef GTEST_HAS_RTTI
// The user didn't tell us whether RTTI is enabled, so we need to
// figure it out.

# ifdef _MSC_VER

#  ifdef _CPPRTTI  // MSVC defines this macro iff RTTI is enabled.
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

// Starting with version 4.3.2, gcc defines __GXX_RTTI iff RTTI is enabled.
# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40302)

#  ifdef __GXX_RTTI
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif  // __GXX_RTTI

// Starting with version 9.0 IBM Visual Age defines __RTTI_ALL__ to 1 if
// both the typeid and dynamic_cast features are present.
# elif defined(__IBMCPP__) && (__IBMCPP__ >= 900)

#  ifdef __RTTI_ALL__
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

# else

// For all other compilers, we assume RTTI is enabled.
#  define GTEST_HAS_RTTI 1

# endif  // _MSC_VER

#endif  // GTEST_HAS_RTTI

// It's this header's responsibility to #include <typeinfo> when RTTI
// is enabled.
#if GTEST_HAS_RTTI
# include <typeinfo>
#endif

// Determines whether Google Test can use the pthreads library.
#ifndef GTEST_HAS_PTHREAD
// The user didn't tell us explicitly, so we assume pthreads support is
// available on Linux and Mac.
//
// To disable threading support in Google Test, add -DGTEST_HAS_PTHREAD=0
// to your compiler flags.
# define GTEST_HAS_PTHREAD (GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_HPUX)
#endif  // GTEST_HAS_PTHREAD

#if GTEST_HAS_PTHREAD
// gtest-port.h guarantees to #include <pthread.h> when GTEST_HAS_PTHREAD is
// true.
# include <pthread.h>  // NOLINT

// For timespec and nanosleep, used below.
# include <time.h>  // NOLINT
#endif

// Determines whether Google Test can use tr1/tuple.  You can define
// this macro to 0 to prevent Google Test from using tuple (any
// feature depending on tuple with be disabled in this mode).
#ifndef GTEST_HAS_TR1_TUPLE
// The user didn't tell us not to do it, so we assume it's OK.
# define GTEST_HAS_TR1_TUPLE 1
#endif  // GTEST_HAS_TR1_TUPLE

// Determines whether Google Test's own tr1 tuple implementation
// should be used.
#ifndef GTEST_USE_OWN_TR1_TUPLE
// The user didn't tell us, so we need to figure it out.

// We use our own TR1 tuple if we aren't sure the user has an
// implementation of it already.  At this time, GCC 4.0.0+ and MSVC
// 2010 are the only mainstream compilers that come with a TR1 tuple
// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
// defining __GNUC__ and friends, but cannot compile GCC's tuple
// implementation.  MSVC 2008 (9.0) provides TR1 tuple in a 323 MB
// Feature Pack download, which we cannot assume the user has.
# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
    || _MSC_VER >= 1600
#  define GTEST_USE_OWN_TR1_TUPLE 0
# else
#  define GTEST_USE_OWN_TR1_TUPLE 1
# endif

#endif  // GTEST_USE_OWN_TR1_TUPLE

// To avoid conditional compilation everywhere, we make it
// gtest-port.h's responsibility to #include the header implementing
// tr1/tuple.
#if GTEST_HAS_TR1_TUPLE

# if GTEST_USE_OWN_TR1_TUPLE
// This file was GENERATED by a script.  DO NOT EDIT BY HAND!!!

// Copyright 2009 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Implements a subset of TR1 tuple needed by Google Test and Google Mock.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_

#include <utility>  // For ::std::pair.

// The compiler used in Symbian has a bug that prevents us from declaring the
// tuple template as a friend (it complains that tuple is redefined).  This
// hack bypasses the bug by declaring the members that should otherwise be
// private as public.
// Sun Studio versions < 12 also have the above bug.
#if defined(__SYMBIAN32__) || (defined(__SUNPRO_CC) && __SUNPRO_CC < 0x590)
# define GTEST_DECLARE_TUPLE_AS_FRIEND_ public:
#else
# define GTEST_DECLARE_TUPLE_AS_FRIEND_ \
    template <GTEST_10_TYPENAMES_(U)> friend class tuple; \
   private:
#endif

// GTEST_n_TUPLE_(T) is the type of an n-tuple.
#define GTEST_0_TUPLE_(T) tuple<>
#define GTEST_1_TUPLE_(T) tuple<T##0, void, void, void, void, void, void, \
    void, void, void>
#define GTEST_2_TUPLE_(T) tuple<T##0, T##1, void, void, void, void, void, \
    void, void, void>
#define GTEST_3_TUPLE_(T) tuple<T##0, T##1, T##2, void, void, void, void, \
    void, void, void>
#define GTEST_4_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, void, void, void, \
    void, void, void>
#define GTEST_5_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, void, void, \
    void, void, void>
#define GTEST_6_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, void, \
    void, void, void>
#define GTEST_7_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    void, void, void>
#define GTEST_8_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, void, void>
#define GTEST_9_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, T##8, void>
#define GTEST_10_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, T##8, T##9>

// GTEST_n_TYPENAMES_(T) declares a list of n typenames.
#define GTEST_0_TYPENAMES_(T)
#define GTEST_1_TYPENAMES_(T) typename T##0
#define GTEST_2_TYPENAMES_(T) typename T##0, typename T##1
#define GTEST_3_TYPENAMES_(T) typename T##0, typename T##1, typename T##2
#define GTEST_4_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3
#define GTEST_5_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4
#define GTEST_6_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5
#define GTEST_7_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6
#define GTEST_8_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, typename T##7
#define GTEST_9_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, \
    typename T##7, typename T##8
#define GTEST_10_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, \
    typename T##7, typename T##8, typename T##9

// In theory, defining stuff in the ::std namespace is undefined
// behavior.  We can do this as we are playing the role of a standard
// library vendor.
namespace std {
namespace tr1 {

template <typename T0 = void, typename T1 = void, typename T2 = void,
    typename T3 = void, typename T4 = void, typename T5 = void,
    typename T6 = void, typename T7 = void, typename T8 = void,
    typename T9 = void>
class tuple;

// Anything in namespace gtest_internal is Google Test's INTERNAL
// IMPLEMENTATION DETAIL and MUST NOT BE USED DIRECTLY in user code.
namespace gtest_internal {

// ByRef<T>::type is T if T is a reference; otherwise it's const T&.
template <typename T>
struct ByRef { typedef const T& type; };  // NOLINT
template <typename T>
struct ByRef<T&> { typedef T& type; };  // NOLINT

// A handy wrapper for ByRef.
#define GTEST_BY_REF_(T) typename ::std::tr1::gtest_internal::ByRef<T>::type

// AddRef<T>::type is T if T is a reference; otherwise it's T&.  This
// is the same as tr1::add_reference<T>::type.
template <typename T>
struct AddRef { typedef T& type; };  // NOLINT
template <typename T>
struct AddRef<T&> { typedef T& type; };  // NOLINT

// A handy wrapper for AddRef.
#define GTEST_ADD_REF_(T) typename ::std::tr1::gtest_internal::AddRef<T>::type

// A helper for implementing get<k>().
template <int k> class Get;

// A helper for implementing tuple_element<k, T>.  kIndexValid is true
// iff k < the number of fields in tuple type T.
template <bool kIndexValid, int kIndex, class Tuple>
struct TupleElement;

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 0, GTEST_10_TUPLE_(T)> { typedef T0 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 1, GTEST_10_TUPLE_(T)> { typedef T1 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 2, GTEST_10_TUPLE_(T)> { typedef T2 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 3, GTEST_10_TUPLE_(T)> { typedef T3 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 4, GTEST_10_TUPLE_(T)> { typedef T4 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 5, GTEST_10_TUPLE_(T)> { typedef T5 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 6, GTEST_10_TUPLE_(T)> { typedef T6 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 7, GTEST_10_TUPLE_(T)> { typedef T7 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 8, GTEST_10_TUPLE_(T)> { typedef T8 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 9, GTEST_10_TUPLE_(T)> { typedef T9 type; };

}  // namespace gtest_internal

template <>
class tuple<> {
 public:
  tuple() {}
  tuple(const tuple& /* t */)  {}
  tuple& operator=(const tuple& /* t */) { return *this; }
};

template <GTEST_1_TYPENAMES_(T)>
class GTEST_1_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0) : f0_(f0) {}

  tuple(const tuple& t) : f0_(t.f0_) {}

  template <GTEST_1_TYPENAMES_(U)>
  tuple(const GTEST_1_TUPLE_(U)& t) : f0_(t.f0_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_1_TYPENAMES_(U)>
  tuple& operator=(const GTEST_1_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_1_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_1_TUPLE_(U)& t) {
    f0_ = t.f0_;
    return *this;
  }

  T0 f0_;
};

template <GTEST_2_TYPENAMES_(T)>
class GTEST_2_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1) : f0_(f0),
      f1_(f1) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_) {}

  template <GTEST_2_TYPENAMES_(U)>
  tuple(const GTEST_2_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_) {}
  template <typename U0, typename U1>
  tuple(const ::std::pair<U0, U1>& p) : f0_(p.first), f1_(p.second) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_2_TYPENAMES_(U)>
  tuple& operator=(const GTEST_2_TUPLE_(U)& t) {
    return CopyFrom(t);
  }
  template <typename U0, typename U1>
  tuple& operator=(const ::std::pair<U0, U1>& p) {
    f0_ = p.first;
    f1_ = p.second;
    return *this;
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_2_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_2_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
};

template <GTEST_3_TYPENAMES_(T)>
class GTEST_3_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2) : f0_(f0), f1_(f1), f2_(f2) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_) {}

  template <GTEST_3_TYPENAMES_(U)>
  tuple(const GTEST_3_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_3_TYPENAMES_(U)>
  tuple& operator=(const GTEST_3_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_3_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_3_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
};

template <GTEST_4_TYPENAMES_(T)>
class GTEST_4_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_) {}

  template <GTEST_4_TYPENAMES_(U)>
  tuple(const GTEST_4_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_4_TYPENAMES_(U)>
  tuple& operator=(const GTEST_4_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_4_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_4_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
};

template <GTEST_5_TYPENAMES_(T)>
class GTEST_5_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3,
      GTEST_BY_REF_(T4) f4) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_) {}

  template <GTEST_5_TYPENAMES_(U)>
  tuple(const GTEST_5_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_5_TYPENAMES_(U)>
  tuple& operator=(const GTEST_5_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_5_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_5_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
};

template <GTEST_6_TYPENAMES_(T)>
class GTEST_6_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_) {}

  template <GTEST_6_TYPENAMES_(U)>
  tuple(const GTEST_6_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_6_TYPENAMES_(U)>
  tuple& operator=(const GTEST_6_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_6_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_6_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
};

template <GTEST_7_TYPENAMES_(T)>
class GTEST_7_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3), f4_(f4), f5_(f5), f6_(f6) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_) {}

  template <GTEST_7_TYPENAMES_(U)>
  tuple(const GTEST_7_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_7_TYPENAMES_(U)>
  tuple& operator=(const GTEST_7_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_7_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_7_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
};

template <GTEST_8_TYPENAMES_(T)>
class GTEST_8_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6,
      GTEST_BY_REF_(T7) f7) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5), f6_(f6), f7_(f7) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_) {}

  template <GTEST_8_TYPENAMES_(U)>
  tuple(const GTEST_8_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_8_TYPENAMES_(U)>
  tuple& operator=(const GTEST_8_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_8_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_8_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
};

template <GTEST_9_TYPENAMES_(T)>
class GTEST_9_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_(), f8_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6, GTEST_BY_REF_(T7) f7,
      GTEST_BY_REF_(T8) f8) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5), f6_(f6), f7_(f7), f8_(f8) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_) {}

  template <GTEST_9_TYPENAMES_(U)>
  tuple(const GTEST_9_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_9_TYPENAMES_(U)>
  tuple& operator=(const GTEST_9_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_9_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_9_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    f8_ = t.f8_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
  T8 f8_;
};

template <GTEST_10_TYPENAMES_(T)>
class tuple {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_(), f8_(),
      f9_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6, GTEST_BY_REF_(T7) f7,
      GTEST_BY_REF_(T8) f8, GTEST_BY_REF_(T9) f9) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3), f4_(f4), f5_(f5), f6_(f6), f7_(f7), f8_(f8), f9_(f9) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_), f9_(t.f9_) {}

  template <GTEST_10_TYPENAMES_(U)>
  tuple(const GTEST_10_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_),
      f9_(t.f9_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_10_TYPENAMES_(U)>
  tuple& operator=(const GTEST_10_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_10_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_10_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    f8_ = t.f8_;
    f9_ = t.f9_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
  T8 f8_;
  T9 f9_;
};

// 6.1.3.2 Tuple creation functions.

// Known limitations: we don't support passing an
// std::tr1::reference_wrapper<T> to make_tuple().  And we don't
// implement tie().

inline tuple<> make_tuple() { return tuple<>(); }

template <GTEST_1_TYPENAMES_(T)>
inline GTEST_1_TUPLE_(T) make_tuple(const T0& f0) {
  return GTEST_1_TUPLE_(T)(f0);
}

template <GTEST_2_TYPENAMES_(T)>
inline GTEST_2_TUPLE_(T) make_tuple(const T0& f0, const T1& f1) {
  return GTEST_2_TUPLE_(T)(f0, f1);
}

template <GTEST_3_TYPENAMES_(T)>
inline GTEST_3_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2) {
  return GTEST_3_TUPLE_(T)(f0, f1, f2);
}

template <GTEST_4_TYPENAMES_(T)>
inline GTEST_4_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3) {
  return GTEST_4_TUPLE_(T)(f0, f1, f2, f3);
}

template <GTEST_5_TYPENAMES_(T)>
inline GTEST_5_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4) {
  return GTEST_5_TUPLE_(T)(f0, f1, f2, f3, f4);
}

template <GTEST_6_TYPENAMES_(T)>
inline GTEST_6_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5) {
  return GTEST_6_TUPLE_(T)(f0, f1, f2, f3, f4, f5);
}

template <GTEST_7_TYPENAMES_(T)>
inline GTEST_7_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6) {
  return GTEST_7_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6);
}

template <GTEST_8_TYPENAMES_(T)>
inline GTEST_8_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7) {
  return GTEST_8_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7);
}

template <GTEST_9_TYPENAMES_(T)>
inline GTEST_9_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7,
    const T8& f8) {
  return GTEST_9_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

template <GTEST_10_TYPENAMES_(T)>
inline GTEST_10_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7,
    const T8& f8, const T9& f9) {
  return GTEST_10_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
}

// 6.1.3.3 Tuple helper classes.

template <typename Tuple> struct tuple_size;

template <GTEST_0_TYPENAMES_(T)>
struct tuple_size<GTEST_0_TUPLE_(T)> { static const int value = 0; };

template <GTEST_1_TYPENAMES_(T)>
struct tuple_size<GTEST_1_TUPLE_(T)> { static const int value = 1; };

template <GTEST_2_TYPENAMES_(T)>
struct tuple_size<GTEST_2_TUPLE_(T)> { static const int value = 2; };

template <GTEST_3_TYPENAMES_(T)>
struct tuple_size<GTEST_3_TUPLE_(T)> { static const int value = 3; };

template <GTEST_4_TYPENAMES_(T)>
struct tuple_size<GTEST_4_TUPLE_(T)> { static const int value = 4; };

template <GTEST_5_TYPENAMES_(T)>
struct tuple_size<GTEST_5_TUPLE_(T)> { static const int value = 5; };

template <GTEST_6_TYPENAMES_(T)>
struct tuple_size<GTEST_6_TUPLE_(T)> { static const int value = 6; };

template <GTEST_7_TYPENAMES_(T)>
struct tuple_size<GTEST_7_TUPLE_(T)> { static const int value = 7; };

template <GTEST_8_TYPENAMES_(T)>
struct tuple_size<GTEST_8_TUPLE_(T)> { static const int value = 8; };

template <GTEST_9_TYPENAMES_(T)>
struct tuple_size<GTEST_9_TUPLE_(T)> { static const int value = 9; };

template <GTEST_10_TYPENAMES_(T)>
struct tuple_size<GTEST_10_TUPLE_(T)> { static const int value = 10; };

template <int k, class Tuple>
struct tuple_element {
  typedef typename gtest_internal::TupleElement<
      k < (tuple_size<Tuple>::value), k, Tuple>::type type;
};

#define GTEST_TUPLE_ELEMENT_(k, Tuple) typename tuple_element<k, Tuple >::type

// 6.1.3.4 Element access.

namespace gtest_internal {

template <>
class Get<0> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(0, Tuple))
  Field(Tuple& t) { return t.f0_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(0, Tuple))
  ConstField(const Tuple& t) { return t.f0_; }
};

template <>
class Get<1> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(1, Tuple))
  Field(Tuple& t) { return t.f1_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(1, Tuple))
  ConstField(const Tuple& t) { return t.f1_; }
};

template <>
class Get<2> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(2, Tuple))
  Field(Tuple& t) { return t.f2_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(2, Tuple))
  ConstField(const Tuple& t) { return t.f2_; }
};

template <>
class Get<3> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(3, Tuple))
  Field(Tuple& t) { return t.f3_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(3, Tuple))
  ConstField(const Tuple& t) { return t.f3_; }
};

template <>
class Get<4> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(4, Tuple))
  Field(Tuple& t) { return t.f4_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(4, Tuple))
  ConstField(const Tuple& t) { return t.f4_; }
};

template <>
class Get<5> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(5, Tuple))
  Field(Tuple& t) { return t.f5_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(5, Tuple))
  ConstField(const Tuple& t) { return t.f5_; }
};

template <>
class Get<6> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(6, Tuple))
  Field(Tuple& t) { return t.f6_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(6, Tuple))
  ConstField(const Tuple& t) { return t.f6_; }
};

template <>
class Get<7> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(7, Tuple))
  Field(Tuple& t) { return t.f7_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(7, Tuple))
  ConstField(const Tuple& t) { return t.f7_; }
};

template <>
class Get<8> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(8, Tuple))
  Field(Tuple& t) { return t.f8_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(8, Tuple))
  ConstField(const Tuple& t) { return t.f8_; }
};

template <>
class Get<9> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(9, Tuple))
  Field(Tuple& t) { return t.f9_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(9, Tuple))
  ConstField(const Tuple& t) { return t.f9_; }
};

}  // namespace gtest_internal

template <int k, GTEST_10_TYPENAMES_(T)>
GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(k, GTEST_10_TUPLE_(T)))
get(GTEST_10_TUPLE_(T)& t) {
  return gtest_internal::Get<k>::Field(t);
}

template <int k, GTEST_10_TYPENAMES_(T)>
GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(k,  GTEST_10_TUPLE_(T)))
get(const GTEST_10_TUPLE_(T)& t) {
  return gtest_internal::Get<k>::ConstField(t);
}

// 6.1.3.5 Relational operators

// We only implement == and !=, as we don't have a need for the rest yet.

namespace gtest_internal {

// SameSizeTuplePrefixComparator<k, k>::Eq(t1, t2) returns true if the
// first k fields of t1 equals the first k fields of t2.
// SameSizeTuplePrefixComparator(k1, k2) would be a compiler error if
// k1 != k2.
template <int kSize1, int kSize2>
struct SameSizeTuplePrefixComparator;

template <>
struct SameSizeTuplePrefixComparator<0, 0> {
  template <class Tuple1, class Tuple2>
  static bool Eq(const Tuple1& /* t1 */, const Tuple2& /* t2 */) {
    return true;
  }
};

template <int k>
struct SameSizeTuplePrefixComparator<k, k> {
  template <class Tuple1, class Tuple2>
  static bool Eq(const Tuple1& t1, const Tuple2& t2) {
    return SameSizeTuplePrefixComparator<k - 1, k - 1>::Eq(t1, t2) &&
        ::std::tr1::get<k - 1>(t1) == ::std::tr1::get<k - 1>(t2);
  }
};

}  // namespace gtest_internal

template <GTEST_10_TYPENAMES_(T), GTEST_10_TYPENAMES_(U)>
inline bool operator==(const GTEST_10_TUPLE_(T)& t,
                       const GTEST_10_TUPLE_(U)& u) {
  return gtest_internal::SameSizeTuplePrefixComparator<
      tuple_size<GTEST_10_TUPLE_(T)>::value,
      tuple_size<GTEST_10_TUPLE_(U)>::value>::Eq(t, u);
}

template <GTEST_10_TYPENAMES_(T), GTEST_10_TYPENAMES_(U)>
inline bool operator!=(const GTEST_10_TUPLE_(T)& t,
                       const GTEST_10_TUPLE_(U)& u) { return !(t == u); }

// 6.1.4 Pairs.
// Unimplemented.

}  // namespace tr1
}  // namespace std

#undef GTEST_0_TUPLE_
#undef GTEST_1_TUPLE_
#undef GTEST_2_TUPLE_
#undef GTEST_3_TUPLE_
#undef GTEST_4_TUPLE_
#undef GTEST_5_TUPLE_
#undef GTEST_6_TUPLE_
#undef GTEST_7_TUPLE_
#undef GTEST_8_TUPLE_
#undef GTEST_9_TUPLE_
#undef GTEST_10_TUPLE_

#undef GTEST_0_TYPENAMES_
#undef GTEST_1_TYPENAMES_
#undef GTEST_2_TYPENAMES_
#undef GTEST_3_TYPENAMES_
#undef GTEST_4_TYPENAMES_
#undef GTEST_5_TYPENAMES_
#undef GTEST_6_TYPENAMES_
#undef GTEST_7_TYPENAMES_
#undef GTEST_8_TYPENAMES_
#undef GTEST_9_TYPENAMES_
#undef GTEST_10_TYPENAMES_

#undef GTEST_DECLARE_TUPLE_AS_FRIEND_
#undef GTEST_BY_REF_
#undef GTEST_ADD_REF_
#undef GTEST_TUPLE_ELEMENT_

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_
# elif GTEST_OS_SYMBIAN

// On Symbian, BOOST_HAS_TR1_TUPLE causes Boost's TR1 tuple library to
// use STLport's tuple implementation, which unfortunately doesn't
// work as the copy of STLport distributed with Symbian is incomplete.
// By making sure BOOST_HAS_TR1_TUPLE is undefined, we force Boost to
// use its own tuple implementation.
#  ifdef BOOST_HAS_TR1_TUPLE
#   undef BOOST_HAS_TR1_TUPLE
#  endif  // BOOST_HAS_TR1_TUPLE

// This prevents <boost/tr1/detail/config.hpp>, which defines
// BOOST_HAS_TR1_TUPLE, from being #included by Boost's <tuple>.
#  define BOOST_TR1_DETAIL_CONFIG_HPP_INCLUDED
#  include <tuple>

# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40000)
// GCC 4.0+ implements tr1/tuple in the <tr1/tuple> header.  This does
// not conform to the TR1 spec, which requires the header to be <tuple>.

#  if !GTEST_HAS_RTTI && GTEST_GCC_VER_ < 40302
// Until version 4.3.2, gcc has a bug that causes <tr1/functional>,
// which is #included by <tr1/tuple>, to not compile when RTTI is
// disabled.  _TR1_FUNCTIONAL is the header guard for
// <tr1/functional>.  Hence the following #define is a hack to prevent
// <tr1/functional> from being included.
#   define _TR1_FUNCTIONAL 1
#   include <tr1/tuple>
#   undef _TR1_FUNCTIONAL  // Allows the user to #include
                        // <tr1/functional> if he chooses to.
#  else
#   include <tr1/tuple>  // NOLINT
#  endif  // !GTEST_HAS_RTTI && GTEST_GCC_VER_ < 40302

# else
// If the compiler is not GCC 4.0+, we assume the user is using a
// spec-conforming TR1 implementation.
#  include <tuple>  // NOLINT
# endif  // GTEST_USE_OWN_TR1_TUPLE

#endif  // GTEST_HAS_TR1_TUPLE

// Determines whether clone(2) is supported.
// Usually it will only be available on Linux, excluding
// Linux on the Itanium architecture.
// Also see http://linux.die.net/man/2/clone.
#ifndef GTEST_HAS_CLONE
// The user didn't tell us, so we need to figure it out.

# if GTEST_OS_LINUX && !defined(__ia64__)
#  define GTEST_HAS_CLONE 1
# else
#  define GTEST_HAS_CLONE 0
# endif  // GTEST_OS_LINUX && !defined(__ia64__)

#endif  // GTEST_HAS_CLONE

// Determines whether to support stream redirection. This is used to test
// output correctness and to implement death tests.
#ifndef GTEST_HAS_STREAM_REDIRECTION
// By default, we assume that stream redirection is supported on all
// platforms except known mobile ones.
# if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_SYMBIAN
#  define GTEST_HAS_STREAM_REDIRECTION 0
# else
#  define GTEST_HAS_STREAM_REDIRECTION 1
# endif  // !GTEST_OS_WINDOWS_MOBILE && !GTEST_OS_SYMBIAN
#endif  // GTEST_HAS_STREAM_REDIRECTION

// Determines whether to support death tests.
// Google Test does not support death tests for VC 7.1 and earlier as
// abort() in a VC 7.1 application compiled as GUI in debug config
// pops up a dialog window that cannot be suppressed programmatically.
#if (GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS || \
     (GTEST_OS_WINDOWS_DESKTOP && _MSC_VER >= 1400) || \
     GTEST_OS_WINDOWS_MINGW || GTEST_OS_AIX || GTEST_OS_HPUX)
# define GTEST_HAS_DEATH_TEST 1
# include <vector>  // NOLINT
#endif

// We don't support MSVC 7.1 with exceptions disabled now.  Therefore
// all the compilers we care about are adequate for supporting
// value-parameterized tests.
#define GTEST_HAS_PARAM_TEST 1

// Determines whether to support type-driven tests.

// Typed tests need <typeinfo> and variadic macros, which GCC, VC++ 8.0,
// Sun Pro CC, IBM Visual Age, and HP aCC support.
#if defined(__GNUC__) || (_MSC_VER >= 1400) || defined(__SUNPRO_CC) || \
    defined(__IBMCPP__) || defined(__HP_aCC)
# define GTEST_HAS_TYPED_TEST 1
# define GTEST_HAS_TYPED_TEST_P 1
#endif

// Determines whether to support Combine(). This only makes sense when
// value-parameterized tests are enabled.  The implementation doesn't
// work on Sun Studio since it doesn't understand templated conversion
// operators.
#if GTEST_HAS_PARAM_TEST && GTEST_HAS_TR1_TUPLE && !defined(__SUNPRO_CC)
# define GTEST_HAS_COMBINE 1
#endif

// Determines whether the system compiler uses UTF-16 for encoding wide strings.
#define GTEST_WIDE_STRING_USES_UTF16_ \
    (GTEST_OS_WINDOWS || GTEST_OS_CYGWIN || GTEST_OS_SYMBIAN || GTEST_OS_AIX)

// Determines whether test results can be streamed to a socket.
#if GTEST_OS_LINUX
# define GTEST_CAN_STREAM_RESULTS_ 1
#endif

// Defines some utility macros.

// The GNU compiler emits a warning if nested "if" statements are followed by
// an "else" statement and braces are not used to explicitly disambiguate the
// "else" binding.  This leads to problems with code like:
//
//   if (gate)
//     ASSERT_*(condition) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#ifdef __INTEL_COMPILER
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_
#else
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0: default:  // NOLINT
#endif

// Use this annotation at the end of a struct/class definition to
// prevent the compiler from optimizing away instances that are never
// used.  This is useful when all interesting logic happens inside the
// c'tor and / or d'tor.  Example:
//
//   struct Foo {
//     Foo() { ... }
//   } GTEST_ATTRIBUTE_UNUSED_;
//
// Also use it after a variable or parameter declaration to tell the
// compiler the variable/parameter does not have to be used.
#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
#else
# define GTEST_ATTRIBUTE_UNUSED_
#endif

// A macro to disallow operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_ASSIGN_(type)\
  void operator=(type const &)

// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)\
  type(type const &);\
  GTEST_DISALLOW_ASSIGN_(type)

// Tell the compiler to warn about unused return values for functions declared
// with this macro.  The macro should be used on function declarations
// following the argument list:
//
//   Sprocket* AllocateSprocket() GTEST_MUST_USE_RESULT_;
#if defined(__GNUC__) && (GTEST_GCC_VER_ >= 30400) && !defined(COMPILER_ICC)
# define GTEST_MUST_USE_RESULT_ __attribute__ ((warn_unused_result))
#else
# define GTEST_MUST_USE_RESULT_
#endif  // __GNUC__ && (GTEST_GCC_VER_ >= 30400) && !COMPILER_ICC

// Determine whether the compiler supports Microsoft's Structured Exception
// Handling.  This is supported by several Windows compilers but generally
// does not exist on any other system.
#ifndef GTEST_HAS_SEH
// The user didn't tell us, so we need to figure it out.

# if defined(_MSC_VER) || defined(__BORLANDC__)
// These two compilers are known to support SEH.
#  define GTEST_HAS_SEH 1
# else
// Assume no SEH.
#  define GTEST_HAS_SEH 0
# endif

#endif  // GTEST_HAS_SEH

#ifdef _MSC_VER

# if GTEST_LINKED_AS_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllimport)
# elif GTEST_CREATE_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllexport)
# endif

#endif  // _MSC_VER

#ifndef GTEST_API_
# define GTEST_API_
#endif

#ifdef __GNUC__
// Ask the compiler to never inline a given function.
# define GTEST_NO_INLINE_ __attribute__((noinline))
#else
# define GTEST_NO_INLINE_
#endif

namespace testing {

class Message;

namespace internal {

class String;

// The GTEST_COMPILE_ASSERT_ macro can be used to verify that a compile time
// expression is true. For example, you could use it to verify the
// size of a static array:
//
//   GTEST_COMPILE_ASSERT_(ARRAYSIZE(content_type_names) == CONTENT_NUM_TYPES,
//                         content_type_names_incorrect_size);
//
// or to make sure a struct is smaller than a certain size:
//
//   GTEST_COMPILE_ASSERT_(sizeof(foo) < 128, foo_too_large);
//
// The second argument to the macro is the name of the variable. If
// the expression is false, most compilers will issue a warning/error
// containing the name of the variable.

template <bool>
struct CompileAssert {
};

#define GTEST_COMPILE_ASSERT_(expr, msg) \
  typedef ::testing::internal::CompileAssert<(bool(expr))> \
      msg[bool(expr) ? 1 : -1]

// Implementation details of GTEST_COMPILE_ASSERT_:
//
// - GTEST_COMPILE_ASSERT_ works by defining an array type that has -1
//   elements (and thus is invalid) when the expression is false.
//
// - The simpler definition
//
//    #define GTEST_COMPILE_ASSERT_(expr, msg) typedef char msg[(expr) ? 1 : -1]
//
//   does not work, as gcc supports variable-length arrays whose sizes
//   are determined at run-time (this is gcc's extension and not part
//   of the C++ standard).  As a result, gcc fails to reject the
//   following code with the simple definition:
//
//     int foo;
//     GTEST_COMPILE_ASSERT_(foo, msg); // not supposed to compile as foo is
//                                      // not a compile-time constant.
//
// - By using the type CompileAssert<(bool(expr))>, we ensures that
//   expr is a compile-time constant.  (Template arguments must be
//   determined at compile-time.)
//
// - The outter parentheses in CompileAssert<(bool(expr))> are necessary
//   to work around a bug in gcc 3.4.4 and 4.0.1.  If we had written
//
//     CompileAssert<bool(expr)>
//
//   instead, these compilers will refuse to compile
//
//     GTEST_COMPILE_ASSERT_(5 > 0, some_message);
//
//   (They seem to think the ">" in "5 > 0" marks the end of the
//   template argument list.)
//
// - The array size is (bool(expr) ? 1 : -1), instead of simply
//
//     ((expr) ? 1 : -1).
//
//   This is to avoid running into a bug in MS VC 7.1, which
//   causes ((0.0) ? 1 : -1) to incorrectly evaluate to 1.

// StaticAssertTypeEqHelper is used by StaticAssertTypeEq defined in gtest.h.
//
// This template is declared, but intentionally undefined.
template <typename T1, typename T2>
struct StaticAssertTypeEqHelper;

template <typename T>
struct StaticAssertTypeEqHelper<T, T> {};

#if GTEST_HAS_GLOBAL_STRING
typedef ::string string;
#else
typedef ::std::string string;
#endif  // GTEST_HAS_GLOBAL_STRING

#if GTEST_HAS_GLOBAL_WSTRING
typedef ::wstring wstring;
#elif GTEST_HAS_STD_WSTRING
typedef ::std::wstring wstring;
#endif  // GTEST_HAS_GLOBAL_WSTRING

// A helper for suppressing warnings on constant condition.  It just
// returns 'condition'.
GTEST_API_ bool IsTrue(bool condition);

// Defines scoped_ptr.

// This implementation of scoped_ptr is PARTIAL - it only contains
// enough stuff to satisfy Google Test's need.
template <typename T>
class scoped_ptr {
 public:
  typedef T element_type;

  explicit scoped_ptr(T* p = NULL) : ptr_(p) {}
  ~scoped_ptr() { reset(); }

  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  T* get() const { return ptr_; }

  T* release() {
    T* const ptr = ptr_;
    ptr_ = NULL;
    return ptr;
  }

  void reset(T* p = NULL) {
    if (p != ptr_) {
      if (IsTrue(sizeof(T) > 0)) {  // Makes sure T is a complete type.
        delete ptr_;
      }
      ptr_ = p;
    }
  }
 private:
  T* ptr_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(scoped_ptr);
};

// Defines RE.

// A simple C++ wrapper for <regex.h>.  It uses the POSIX Extended
// Regular Expression syntax.
class GTEST_API_ RE {
 public:
  // A copy constructor is required by the Standard to initialize object
  // references from r-values.
  RE(const RE& other) { Init(other.pattern()); }

  // Constructs an RE from a string.
  RE(const ::std::string& regex) { Init(regex.c_str()); }  // NOLINT

#if GTEST_HAS_GLOBAL_STRING

  RE(const ::string& regex) { Init(regex.c_str()); }  // NOLINT

#endif  // GTEST_HAS_GLOBAL_STRING

  RE(const char* regex) { Init(regex); }  // NOLINT
  ~RE();

  // Returns the string representation of the regex.
  const char* pattern() const { return pattern_; }

  // FullMatch(str, re) returns true iff regular expression re matches
  // the entire str.
  // PartialMatch(str, re) returns true iff regular expression re
  // matches a substring of str (including str itself).
  //
  // TODO(wan@google.com): make FullMatch() and PartialMatch() work
  // when str contains NUL characters.
  static bool FullMatch(const ::std::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::std::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }

#if GTEST_HAS_GLOBAL_STRING

  static bool FullMatch(const ::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }

#endif  // GTEST_HAS_GLOBAL_STRING

  static bool FullMatch(const char* str, const RE& re);
  static bool PartialMatch(const char* str, const RE& re);

 private:
  void Init(const char* regex);

  // We use a const char* instead of a string, as Google Test may be used
  // where string is not available.  We also do not use Google Test's own
  // String type here, in order to simplify dependencies between the
  // files.
  const char* pattern_;
  bool is_valid_;

#if GTEST_USES_POSIX_RE

  regex_t full_regex_;     // For FullMatch().
  regex_t partial_regex_;  // For PartialMatch().

#else  // GTEST_USES_SIMPLE_RE

  const char* full_pattern_;  // For FullMatch();

#endif

  GTEST_DISALLOW_ASSIGN_(RE);
};

// Formats a source file path and a line number as they would appear
// in an error message from the compiler used to compile this code.
GTEST_API_ ::std::string FormatFileLocation(const char* file, int line);

// Formats a file location for compiler-independent XML output.
// Although this function is not platform dependent, we put it next to
// FormatFileLocation in order to contrast the two functions.
GTEST_API_ ::std::string FormatCompilerIndependentFileLocation(const char* file,
                                                               int line);

// Defines logging utilities:
//   GTEST_LOG_(severity) - logs messages at the specified severity level. The
//                          message itself is streamed into the macro.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.

enum GTestLogSeverity {
  GTEST_INFO,
  GTEST_WARNING,
  GTEST_ERROR,
  GTEST_FATAL
};

// Formats log entry severity, provides a stream object for streaming the
// log message, and terminates the message with a newline when going out of
// scope.
class GTEST_API_ GTestLog {
 public:
  GTestLog(GTestLogSeverity severity, const char* file, int line);

  // Flushes the buffers and, if severity is GTEST_FATAL, aborts the program.
  ~GTestLog();

  ::std::ostream& GetStream() { return ::std::cerr; }

 private:
  const GTestLogSeverity severity_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestLog);
};

#define GTEST_LOG_(severity) \
    ::testing::internal::GTestLog(::testing::internal::GTEST_##severity, \
                                  __FILE__, __LINE__).GetStream()

inline void LogToStderr() {}
inline void FlushInfoLog() { fflush(NULL); }

// INTERNAL IMPLEMENTATION - DO NOT USE.
//
// GTEST_CHECK_ is an all-mode assert. It aborts the program if the condition
// is not satisfied.
//  Synopsys:
//    GTEST_CHECK_(boolean_condition);
//     or
//    GTEST_CHECK_(boolean_condition) << "Additional message";
//
//    This checks the condition and if the condition is not satisfied
//    it prints message about the condition violation, including the
//    condition itself, plus additional message streamed into it, if any,
//    and then it aborts the program. It aborts the program irrespective of
//    whether it is built in the debug mode or not.
#define GTEST_CHECK_(condition) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    if (::testing::internal::IsTrue(condition)) \
      ; \
    else \
      GTEST_LOG_(FATAL) << "Condition " #condition " failed. "

// An all-mode assert to verify that the given POSIX-style function
// call returns 0 (indicating success).  Known limitation: this
// doesn't expand to a balanced 'if' statement, so enclose the macro
// in {} if you need to use it as the only statement in an 'if'
// branch.
#define GTEST_CHECK_POSIX_SUCCESS_(posix_call) \
  if (const int gtest_error = (posix_call)) \
    GTEST_LOG_(FATAL) << #posix_call << "failed with error " \
                      << gtest_error

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Use ImplicitCast_ as a safe version of static_cast for upcasting in
// the type hierarchy (e.g. casting a Foo* to a SuperclassOfFoo* or a
// const Foo*).  When you use ImplicitCast_, the compiler checks that
// the cast is safe.  Such explicit ImplicitCast_s are necessary in
// surprisingly many situations where C++ demands an exact type match
// instead of an argument type convertable to a target type.
//
// The syntax for using ImplicitCast_ is the same as for static_cast:
//
//   ImplicitCast_<ToType>(expr)
//
// ImplicitCast_ would have been part of the C++ standard library,
// but the proposal was submitted too late.  It will probably make
// its way into the language in the future.
//
// This relatively ugly name is intentional. It prevents clashes with
// similar functions users may have (e.g., implicit_cast). The internal
// namespace alone is not enough because the function can be found by ADL.
template<typename To>
inline To ImplicitCast_(To x) { return x; }

// When you upcast (that is, cast a pointer from type Foo to type
// SuperclassOfFoo), it's fine to use ImplicitCast_<>, since upcasts
// always succeed.  When you downcast (that is, cast a pointer from
// type Foo to type SubclassOfFoo), static_cast<> isn't safe, because
// how do you know the pointer is really of type SubclassOfFoo?  It
// could be a bare Foo, or of type DifferentSubclassOfFoo.  Thus,
// when you downcast, you should use this macro.  In debug mode, we
// use dynamic_cast<> to double-check the downcast is legal (we die
// if it's not).  In normal mode, we do the efficient static_cast<>
// instead.  Thus, it's important to test in debug mode to make sure
// the cast is legal!
//    This is the only place in the code we should use dynamic_cast<>.
// In particular, you SHOULDN'T be using dynamic_cast<> in order to
// do RTTI (eg code like this:
//    if (dynamic_cast<Subclass1>(foo)) HandleASubclass1Object(foo);
//    if (dynamic_cast<Subclass2>(foo)) HandleASubclass2Object(foo);
// You should design the code some other way not to need this.
//
// This relatively ugly name is intentional. It prevents clashes with
// similar functions users may have (e.g., down_cast). The internal
// namespace alone is not enough because the function can be found by ADL.
template<typename To, typename From>  // use like this: DownCast_<T*>(foo);
inline To DownCast_(From* f) {  // so we only accept pointers
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    const To to = NULL;
    ::testing::internal::ImplicitCast_<From*>(to);
  }

#if GTEST_HAS_RTTI
  // RTTI: debug mode only!
  GTEST_CHECK_(f == NULL || dynamic_cast<To>(f) != NULL);
#endif
  return static_cast<To>(f);
}

// Downcasts the pointer of type Base to Derived.
// Derived must be a subclass of Base. The parameter MUST
// point to a class of type Derived, not any subclass of it.
// When RTTI is available, the function performs a runtime
// check to enforce this.
template <class Derived, class Base>
Derived* CheckedDowncastToActualType(Base* base) {
#if GTEST_HAS_RTTI
  GTEST_CHECK_(typeid(*base) == typeid(Derived));
  return dynamic_cast<Derived*>(base);  // NOLINT
#else
  return static_cast<Derived*>(base);  // Poor man's downcast.
#endif
}

#if GTEST_HAS_STREAM_REDIRECTION

// Defines the stderr capturer:
//   CaptureStdout     - starts capturing stdout.
//   GetCapturedStdout - stops capturing stdout and returns the captured string.
//   CaptureStderr     - starts capturing stderr.
//   GetCapturedStderr - stops capturing stderr and returns the captured string.
//
GTEST_API_ void CaptureStdout();
GTEST_API_ String GetCapturedStdout();
GTEST_API_ void CaptureStderr();
GTEST_API_ String GetCapturedStderr();

#endif  // GTEST_HAS_STREAM_REDIRECTION


#if GTEST_HAS_DEATH_TEST

// A copy of all command line arguments.  Set by InitGoogleTest().
extern ::std::vector<String> g_argvs;

// GTEST_HAS_DEATH_TEST implies we have ::std::string.
const ::std::vector<String>& GetArgvs();

#endif  // GTEST_HAS_DEATH_TEST

// Defines synchronization primitives.

#if GTEST_HAS_PTHREAD

// Sleeps for (roughly) n milli-seconds.  This function is only for
// testing Google Test's own constructs.  Don't use it in user tests,
// either directly or indirectly.
inline void SleepMilliseconds(int n) {
  const timespec time = {
    0,                  // 0 seconds.
    n * 1000L * 1000L,  // And n ms.
  };
  nanosleep(&time, NULL);
}

// Allows a controller thread to pause execution of newly created
// threads until notified.  Instances of this class must be created
// and destroyed in the controller thread.
//
// This class is only for testing Google Test's own constructs. Do not
// use it in user tests, either directly or indirectly.
class Notification {
 public:
  Notification() : notified_(false) {}

  // Notifies all threads created with this notification to start. Must
  // be called from the controller thread.
  void Notify() { notified_ = true; }

  // Blocks until the controller thread notifies. Must be called from a test
  // thread.
  void WaitForNotification() {
    while(!notified_) {
      SleepMilliseconds(10);
    }
  }

 private:
  volatile bool notified_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(Notification);
};

// As a C-function, ThreadFuncWithCLinkage cannot be templated itself.
// Consequently, it cannot select a correct instantiation of ThreadWithParam
// in order to call its Run(). Introducing ThreadWithParamBase as a
// non-templated base class for ThreadWithParam allows us to bypass this
// problem.
class ThreadWithParamBase {
 public:
  virtual ~ThreadWithParamBase() {}
  virtual void Run() = 0;
};

// pthread_create() accepts a pointer to a function type with the C linkage.
// According to the Standard (7.5/1), function types with different linkages
// are different even if they are otherwise identical.  Some compilers (for
// example, SunStudio) treat them as different types.  Since class methods
// cannot be defined with C-linkage we need to define a free C-function to
// pass into pthread_create().
extern "C" inline void* ThreadFuncWithCLinkage(void* thread) {
  static_cast<ThreadWithParamBase*>(thread)->Run();
  return NULL;
}

// Helper class for testing Google Test's multi-threading constructs.
// To use it, write:
//
//   void ThreadFunc(int param) { /* Do things with param */ }
//   Notification thread_can_start;
//   ...
//   // The thread_can_start parameter is optional; you can supply NULL.
//   ThreadWithParam<int> thread(&ThreadFunc, 5, &thread_can_start);
//   thread_can_start.Notify();
//
// These classes are only for testing Google Test's own constructs. Do
// not use them in user tests, either directly or indirectly.
template <typename T>
class ThreadWithParam : public ThreadWithParamBase {
 public:
  typedef void (*UserThreadFunc)(T);

  ThreadWithParam(
      UserThreadFunc func, T param, Notification* thread_can_start)
      : func_(func),
        param_(param),
        thread_can_start_(thread_can_start),
        finished_(false) {
    ThreadWithParamBase* const base = this;
    // The thread can be created only after all fields except thread_
    // have been initialized.
    GTEST_CHECK_POSIX_SUCCESS_(
        pthread_create(&thread_, 0, &ThreadFuncWithCLinkage, base));
  }
  ~ThreadWithParam() { Join(); }

  void Join() {
    if (!finished_) {
      GTEST_CHECK_POSIX_SUCCESS_(pthread_join(thread_, 0));
      finished_ = true;
    }
  }

  virtual void Run() {
    if (thread_can_start_ != NULL)
      thread_can_start_->WaitForNotification();
    func_(param_);
  }

 private:
  const UserThreadFunc func_;  // User-supplied thread function.
  const T param_;  // User-supplied parameter to the thread function.
  // When non-NULL, used to block execution until the controller thread
  // notifies.
  Notification* const thread_can_start_;
  bool finished_;  // true iff we know that the thread function has finished.
  pthread_t thread_;  // The native thread object.

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadWithParam);
};

// MutexBase and Mutex implement mutex on pthreads-based platforms. They
// are used in conjunction with class MutexLock:
//
//   Mutex mutex;
//   ...
//   MutexLock lock(&mutex);  // Acquires the mutex and releases it at the end
//                            // of the current scope.
//
// MutexBase implements behavior for both statically and dynamically
// allocated mutexes.  Do not use MutexBase directly.  Instead, write
// the following to define a static mutex:
//
//   GTEST_DEFINE_STATIC_MUTEX_(g_some_mutex);
//
// You can forward declare a static mutex like this:
//
//   GTEST_DECLARE_STATIC_MUTEX_(g_some_mutex);
//
// To create a dynamic mutex, just define an object of type Mutex.
class MutexBase {
 public:
  // Acquires this mutex.
  void Lock() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_lock(&mutex_));
    owner_ = pthread_self();
  }

  // Releases this mutex.
  void Unlock() {
    // We don't protect writing to owner_ here, as it's the caller's
    // responsibility to ensure that the current thread holds the
    // mutex when this is called.
    owner_ = 0;
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_unlock(&mutex_));
  }

  // Does nothing if the current thread holds the mutex. Otherwise, crashes
  // with high probability.
  void AssertHeld() const {
    GTEST_CHECK_(owner_ == pthread_self())
        << "The current thread is not holding the mutex @" << this;
  }

  // A static mutex may be used before main() is entered.  It may even
  // be used before the dynamic initialization stage.  Therefore we
  // must be able to initialize a static mutex object at link time.
  // This means MutexBase has to be a POD and its member variables
  // have to be public.
 public:
  pthread_mutex_t mutex_;  // The underlying pthread mutex.
  pthread_t owner_;  // The thread holding the mutex; 0 means no one holds it.
};

// Forward-declares a static mutex.
# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
    extern ::testing::internal::MutexBase mutex

// Defines and statically (i.e. at link time) initializes a static mutex.
# define GTEST_DEFINE_STATIC_MUTEX_(mutex) \
    ::testing::internal::MutexBase mutex = { PTHREAD_MUTEX_INITIALIZER, 0 }

// The Mutex class can only be used for mutexes created at runtime. It
// shares its API with MutexBase otherwise.
class Mutex : public MutexBase {
 public:
  Mutex() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, NULL));
    owner_ = 0;
  }
  ~Mutex() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_destroy(&mutex_));
  }

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(Mutex);
};

// We cannot name this class MutexLock as the ctor declaration would
// conflict with a macro named MutexLock, which is defined on some
// platforms.  Hence the typedef trick below.
class GTestMutexLock {
 public:
  explicit GTestMutexLock(MutexBase* mutex)
      : mutex_(mutex) { mutex_->Lock(); }

  ~GTestMutexLock() { mutex_->Unlock(); }

 private:
  MutexBase* const mutex_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestMutexLock);
};

typedef GTestMutexLock MutexLock;

// Helpers for ThreadLocal.

// pthread_key_create() requires DeleteThreadLocalValue() to have
// C-linkage.  Therefore it cannot be templatized to access
// ThreadLocal<T>.  Hence the need for class
// ThreadLocalValueHolderBase.
class ThreadLocalValueHolderBase {
 public:
  virtual ~ThreadLocalValueHolderBase() {}
};

// Called by pthread to delete thread-local data stored by
// pthread_setspecific().
extern "C" inline void DeleteThreadLocalValue(void* value_holder) {
  delete static_cast<ThreadLocalValueHolderBase*>(value_holder);
}

// Implements thread-local storage on pthreads-based systems.
//
//   // Thread 1
//   ThreadLocal<int> tl(100);  // 100 is the default value for each thread.
//
//   // Thread 2
//   tl.set(150);  // Changes the value for thread 2 only.
//   EXPECT_EQ(150, tl.get());
//
//   // Thread 1
//   EXPECT_EQ(100, tl.get());  // In thread 1, tl has the original value.
//   tl.set(200);
//   EXPECT_EQ(200, tl.get());
//
// The template type argument T must have a public copy constructor.
// In addition, the default ThreadLocal constructor requires T to have
// a public default constructor.
//
// An object managed for a thread by a ThreadLocal instance is deleted
// when the thread exits.  Or, if the ThreadLocal instance dies in
// that thread, when the ThreadLocal dies.  It's the user's
// responsibility to ensure that all other threads using a ThreadLocal
// have exited when it dies, or the per-thread objects for those
// threads will not be deleted.
//
// Google Test only uses global ThreadLocal objects.  That means they
// will die after main() has returned.  Therefore, no per-thread
// object managed by Google Test will be leaked as long as all threads
// using Google Test have exited when main() returns.
template <typename T>
class ThreadLocal {
 public:
  ThreadLocal() : key_(CreateKey()),
                  default_() {}
  explicit ThreadLocal(const T& value) : key_(CreateKey()),
                                         default_(value) {}

  ~ThreadLocal() {
    // Destroys the managed object for the current thread, if any.
    DeleteThreadLocalValue(pthread_getspecific(key_));

    // Releases resources associated with the key.  This will *not*
    // delete managed objects for other threads.
    GTEST_CHECK_POSIX_SUCCESS_(pthread_key_delete(key_));
  }

  T* pointer() { return GetOrCreateValue(); }
  const T* pointer() const { return GetOrCreateValue(); }
  const T& get() const { return *pointer(); }
  void set(const T& value) { *pointer() = value; }

 private:
  // Holds a value of type T.
  class ValueHolder : public ThreadLocalValueHolderBase {
   public:
    explicit ValueHolder(const T& value) : value_(value) {}

    T* pointer() { return &value_; }

   private:
    T value_;
    GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolder);
  };

  static pthread_key_t CreateKey() {
    pthread_key_t key;
    // When a thread exits, DeleteThreadLocalValue() will be called on
    // the object managed for that thread.
    GTEST_CHECK_POSIX_SUCCESS_(
        pthread_key_create(&key, &DeleteThreadLocalValue));
    return key;
  }

  T* GetOrCreateValue() const {
    ThreadLocalValueHolderBase* const holder =
        static_cast<ThreadLocalValueHolderBase*>(pthread_getspecific(key_));
    if (holder != NULL) {
      return CheckedDowncastToActualType<ValueHolder>(holder)->pointer();
    }

    ValueHolder* const new_holder = new ValueHolder(default_);
    ThreadLocalValueHolderBase* const holder_base = new_holder;
    GTEST_CHECK_POSIX_SUCCESS_(pthread_setspecific(key_, holder_base));
    return new_holder->pointer();
  }

  // A key pthreads uses for looking up per-thread values.
  const pthread_key_t key_;
  const T default_;  // The default value for each thread.

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocal);
};

# define GTEST_IS_THREADSAFE 1

#else  // GTEST_HAS_PTHREAD

// A dummy implementation of synchronization primitives (mutex, lock,
// and thread-local variable).  Necessary for compiling Google Test where
// mutex is not supported - using Google Test in multiple threads is not
// supported on such platforms.

class Mutex {
 public:
  Mutex() {}
  void AssertHeld() const {}
};

# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
  extern ::testing::internal::Mutex mutex

# define GTEST_DEFINE_STATIC_MUTEX_(mutex) ::testing::internal::Mutex mutex

class GTestMutexLock {
 public:
  explicit GTestMutexLock(Mutex*) {}  // NOLINT
};

typedef GTestMutexLock MutexLock;

template <typename T>
class ThreadLocal {
 public:
  ThreadLocal() : value_() {}
  explicit ThreadLocal(const T& value) : value_(value) {}
  T* pointer() { return &value_; }
  const T* pointer() const { return &value_; }
  const T& get() const { return value_; }
  void set(const T& value) { value_ = value; }
 private:
  T value_;
};

// The above synchronization primitives have dummy implementations.
// Therefore Google Test is not thread-safe.
# define GTEST_IS_THREADSAFE 0

#endif  // GTEST_HAS_PTHREAD

// Returns the number of threads running in the process, or 0 to indicate that
// we cannot detect it.
GTEST_API_ size_t GetThreadCount();

// Passing non-POD classes through ellipsis (...) crashes the ARM
// compiler and generates a warning in Sun Studio.  The Nokia Symbian
// and the IBM XL C/C++ compiler try to instantiate a copy constructor
// for objects passed through ellipsis (...), failing for uncopyable
// objects.  We define this to ensure that only POD is passed through
// ellipsis on these systems.
#if defined(__SYMBIAN32__) || defined(__IBMCPP__) || defined(__SUNPRO_CC)
// We lose support for NULL detection where the compiler doesn't like
// passing non-POD classes through ellipsis (...).
# define GTEST_ELLIPSIS_NEEDS_POD_ 1
#else
# define GTEST_CAN_COMPARE_NULL 1
#endif

// The Nokia Symbian and IBM XL C/C++ compilers cannot decide between
// const T& and const T* in a function template.  These compilers
// _can_ decide between class template specializations for T and T*,
// so a tr1::type_traits-like is_pointer works.
#if defined(__SYMBIAN32__) || defined(__IBMCPP__)
# define GTEST_NEEDS_IS_POINTER_ 1
#endif

template <bool bool_value>
struct bool_constant {
  typedef bool_constant<bool_value> type;
  static const bool value = bool_value;
};
template <bool bool_value> const bool bool_constant<bool_value>::value;

typedef bool_constant<false> false_type;
typedef bool_constant<true> true_type;

template <typename T>
struct is_pointer : public false_type {};

template <typename T>
struct is_pointer<T*> : public true_type {};

template <typename Iterator>
struct IteratorTraits {
  typedef typename Iterator::value_type value_type;
};

template <typename T>
struct IteratorTraits<T*> {
  typedef T value_type;
};

template <typename T>
struct IteratorTraits<const T*> {
  typedef T value_type;
};

#if GTEST_OS_WINDOWS
# define GTEST_PATH_SEP_ "\\"
# define GTEST_HAS_ALT_PATH_SEP_ 1
// The biggest signed integer type the compiler supports.
typedef __int64 BiggestInt;
#else
# define GTEST_PATH_SEP_ "/"
# define GTEST_HAS_ALT_PATH_SEP_ 0
typedef long long BiggestInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS

// Utilities for char.

// isspace(int ch) and friends accept an unsigned char or EOF.  char
// may be signed, depending on the compiler (or compiler flags).
// Therefore we need to cast a char to unsigned char before calling
// isspace(), etc.

inline bool IsAlpha(char ch) {
  return isalpha(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsAlNum(char ch) {
  return isalnum(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsDigit(char ch) {
  return isdigit(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsLower(char ch) {
  return islower(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsSpace(char ch) {
  return isspace(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsUpper(char ch) {
  return isupper(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsXDigit(char ch) {
  return isxdigit(static_cast<unsigned char>(ch)) != 0;
}

inline char ToLower(char ch) {
  return static_cast<char>(tolower(static_cast<unsigned char>(ch)));
}
inline char ToUpper(char ch) {
  return static_cast<char>(toupper(static_cast<unsigned char>(ch)));
}

// The testing::internal::posix namespace holds wrappers for common
// POSIX functions.  These wrappers hide the differences between
// Windows/MSVC and POSIX systems.  Since some compilers define these
// standard functions as macros, the wrapper cannot have the same name
// as the wrapped function.

namespace posix {

// Functions with a different name on Windows.

#if GTEST_OS_WINDOWS

typedef struct _stat StatStruct;

# ifdef __BORLANDC__
inline int IsATTY(int fd) { return isatty(fd); }
inline int StrCaseCmp(const char* s1, const char* s2) {
  return stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
# else  // !__BORLANDC__
#  if GTEST_OS_WINDOWS_MOBILE
inline int IsATTY(int /* fd */) { return 0; }
#  else
inline int IsATTY(int fd) { return _isatty(fd); }
#  endif  // GTEST_OS_WINDOWS_MOBILE
inline int StrCaseCmp(const char* s1, const char* s2) {
  return _stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return _strdup(src); }
# endif  // __BORLANDC__

# if GTEST_OS_WINDOWS_MOBILE
inline int FileNo(FILE* file) { return reinterpret_cast<int>(_fileno(file)); }
// Stat(), RmDir(), and IsDir() are not needed on Windows CE at this
// time and thus not defined there.
# else
inline int FileNo(FILE* file) { return _fileno(file); }
inline int Stat(const char* path, StatStruct* buf) { return _stat(path, buf); }
inline int RmDir(const char* dir) { return _rmdir(dir); }
inline bool IsDir(const StatStruct& st) {
  return (_S_IFDIR & st.st_mode) != 0;
}
# endif  // GTEST_OS_WINDOWS_MOBILE

#else

typedef struct stat StatStruct;

inline int FileNo(FILE* file) { return fileno(file); }
inline int IsATTY(int fd) { return isatty(fd); }
inline int Stat(const char* path, StatStruct* buf) { return stat(path, buf); }
inline int StrCaseCmp(const char* s1, const char* s2) {
  return strcasecmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
inline int RmDir(const char* dir) { return rmdir(dir); }
inline bool IsDir(const StatStruct& st) { return S_ISDIR(st.st_mode); }

#endif  // GTEST_OS_WINDOWS

// Functions deprecated by MSVC 8.0.

#ifdef _MSC_VER
// Temporarily disable warning 4996 (deprecated function).
# pragma warning(push)
# pragma warning(disable:4996)
#endif

inline const char* StrNCpy(char* dest, const char* src, size_t n) {
  return strncpy(dest, src, n);
}

// ChDir(), FReopen(), FDOpen(), Read(), Write(), Close(), and
// StrError() aren't needed on Windows CE at this time and thus not
// defined there.

#if !GTEST_OS_WINDOWS_MOBILE
inline int ChDir(const char* dir) { return chdir(dir); }
#endif
inline FILE* FOpen(const char* path, const char* mode) {
  return fopen(path, mode);
}
#if !GTEST_OS_WINDOWS_MOBILE
inline FILE *FReopen(const char* path, const char* mode, FILE* stream) {
  return freopen(path, mode, stream);
}
inline FILE* FDOpen(int fd, const char* mode) { return fdopen(fd, mode); }
#endif
inline int FClose(FILE* fp) { return fclose(fp); }
#if !GTEST_OS_WINDOWS_MOBILE
inline int Read(int fd, void* buf, unsigned int count) {
  return static_cast<int>(read(fd, buf, count));
}
inline int Write(int fd, const void* buf, unsigned int count) {
  return static_cast<int>(write(fd, buf, count));
}
inline int Close(int fd) { return close(fd); }
inline const char* StrError(int errnum) { return strerror(errnum); }
#endif
inline const char* GetEnv(const char* name) {
#if GTEST_OS_WINDOWS_MOBILE
  // We are on Windows CE, which has no environment variables.
  return NULL;
#elif defined(__BORLANDC__) || defined(__SunOS_5_8) || defined(__SunOS_5_9)
  // Environment variables which we programmatically clear will be set to the
  // empty string rather than unset (NULL).  Handle that case.
  const char* const env = getenv(name);
  return (env != NULL && env[0] != '\0') ? env : NULL;
#else
  return getenv(name);
#endif
}

#ifdef _MSC_VER
# pragma warning(pop)  // Restores the warning state.
#endif

#if GTEST_OS_WINDOWS_MOBILE
// Windows CE has no C library. The abort() function is used in
// several places in Google Test. This implementation provides a reasonable
// imitation of standard behaviour.
void Abort();
#else
inline void Abort() { abort(); }
#endif  // GTEST_OS_WINDOWS_MOBILE

}  // namespace posix

// The maximum number a BiggestInt can represent.  This definition
// works no matter BiggestInt is represented in one's complement or
// two's complement.
//
// We cannot rely on numeric_limits in STL, as __int64 and long long
// are not part of standard C++ and numeric_limits doesn't need to be
// defined for them.
const BiggestInt kMaxBiggestInt =
    ~(static_cast<BiggestInt>(1) << (8*sizeof(BiggestInt) - 1));

// This template class serves as a compile-time function from size to
// type.  It maps a size in bytes to a primitive type with that
// size. e.g.
//
//   TypeWithSize<4>::UInt
//
// is typedef-ed to be unsigned int (unsigned integer made up of 4
// bytes).
//
// Such functionality should belong to STL, but I cannot find it
// there.
//
// Google Test uses this class in the implementation of floating-point
// comparison.
//
// For now it only handles UInt (unsigned int) as that's all Google Test
// needs.  Other types can be easily added in the future if need
// arises.
template <size_t size>
class TypeWithSize {
 public:
  // This prevents the user from using TypeWithSize<N> with incorrect
  // values of N.
  typedef void UInt;
};

// The specialization for size 4.
template <>
class TypeWithSize<4> {
 public:
  // unsigned int has size 4 in both gcc and MSVC.
  //
  // As base/basictypes.h doesn't compile on Windows, we cannot use
  // uint32, uint64, and etc here.
  typedef int Int;
  typedef unsigned int UInt;
};

// The specialization for size 8.
template <>
class TypeWithSize<8> {
 public:

#if GTEST_OS_WINDOWS
  typedef __int64 Int;
  typedef unsigned __int64 UInt;
#else
  typedef long long Int;  // NOLINT
  typedef unsigned long long UInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS
};

// Integer types of known sizes.
typedef TypeWithSize<4>::Int Int32;
typedef TypeWithSize<4>::UInt UInt32;
typedef TypeWithSize<8>::Int Int64;
typedef TypeWithSize<8>::UInt UInt64;
typedef TypeWithSize<8>::Int TimeInMillis;  // Represents time in milliseconds.

// Utilities for command line flags and environment variables.

// Macro for referencing flags.
#define GTEST_FLAG(name) FLAGS_gtest_##name

// Macros for declaring flags.
#define GTEST_DECLARE_bool_(name) GTEST_API_ extern bool GTEST_FLAG(name)
#define GTEST_DECLARE_int32_(name) \
    GTEST_API_ extern ::testing::internal::Int32 GTEST_FLAG(name)
#define GTEST_DECLARE_string_(name) \
    GTEST_API_ extern ::testing::internal::String GTEST_FLAG(name)

// Macros for defining flags.
#define GTEST_DEFINE_bool_(name, default_val, doc) \
    GTEST_API_ bool GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_int32_(name, default_val, doc) \
    GTEST_API_ ::testing::internal::Int32 GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_string_(name, default_val, doc) \
    GTEST_API_ ::testing::internal::String GTEST_FLAG(name) = (default_val)

// Parses 'str' for a 32-bit signed integer.  If successful, writes the result
// to *value and returns true; otherwise leaves *value unchanged and returns
// false.
// TODO(chandlerc): Find a better way to refactor flag and environment parsing
// out of both gtest-port.cc and gtest.cc to avoid exporting this utility
// function.
bool ParseInt32(const Message& src_text, const char* str, Int32* value);

// Parses a bool/Int32/string from the environment variable
// corresponding to the given Google Test flag.
bool BoolFromGTestEnv(const char* flag, bool default_val);
GTEST_API_ Int32 Int32FromGTestEnv(const char* flag, Int32 default_val);
const char* StringFromGTestEnv(const char* flag, const char* default_val);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

#if GTEST_OS_LINUX
# include <stdlib.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <unistd.h>
#endif  // GTEST_OS_LINUX

#include <ctype.h>
#include <string.h>
#include <iomanip>
#include <limits>
#include <set>

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares the String class and functions used internally by
// Google Test.  They are subject to change without notice. They should not used
// by code external to Google Test.
//
// This header file is #included by <gtest/internal/gtest-internal.h>.
// It should not be #included by other files.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_

#ifdef __BORLANDC__
// string.h is not guaranteed to provide strcpy on C++ Builder.
# include <mem.h>
#endif

#include <string.h>

#include <string>

namespace testing {
namespace internal {

// String - a UTF-8 string class.
//
// For historic reasons, we don't use std::string.
//
// TODO(wan@google.com): replace this class with std::string or
// implement it in terms of the latter.
//
// Note that String can represent both NULL and the empty string,
// while std::string cannot represent NULL.
//
// NULL and the empty string are considered different.  NULL is less
// than anything (including the empty string) except itself.
//
// This class only provides minimum functionality necessary for
// implementing Google Test.  We do not intend to implement a full-fledged
// string class here.
//
// Since the purpose of this class is to provide a substitute for
// std::string on platforms where it cannot be used, we define a copy
// constructor and assignment operators such that we don't need
// conditional compilation in a lot of places.
//
// In order to make the representation efficient, the d'tor of String
// is not virtual.  Therefore DO NOT INHERIT FROM String.
class GTEST_API_ String {
 public:
  // Static utility methods

  // Returns the input enclosed in double quotes if it's not NULL;
  // otherwise returns "(null)".  For example, "\"Hello\"" is returned
  // for input "Hello".
  //
  // This is useful for printing a C string in the syntax of a literal.
  //
  // Known issue: escape sequences are not handled yet.
  static String ShowCStringQuoted(const char* c_str);

  // Clones a 0-terminated C string, allocating memory using new.  The
  // caller is responsible for deleting the return value using
  // delete[].  Returns the cloned string, or NULL if the input is
  // NULL.
  //
  // This is different from strdup() in string.h, which allocates
  // memory using malloc().
  static const char* CloneCString(const char* c_str);

#if GTEST_OS_WINDOWS_MOBILE
  // Windows CE does not have the 'ANSI' versions of Win32 APIs. To be
  // able to pass strings to Win32 APIs on CE we need to convert them
  // to 'Unicode', UTF-16.

  // Creates a UTF-16 wide string from the given ANSI string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the wide string, or NULL if the
  // input is NULL.
  //
  // The wide string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static LPCWSTR AnsiToUtf16(const char* c_str);

  // Creates an ANSI string from the given wide string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the ANSI string, or NULL if the
  // input is NULL.
  //
  // The returned string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static const char* Utf16ToAnsi(LPCWSTR utf16_str);
#endif

  // Compares two C strings.  Returns true iff they have the same content.
  //
  // Unlike strcmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CStringEquals(const char* lhs, const char* rhs);

  // Converts a wide C string to a String using the UTF-8 encoding.
  // NULL will be converted to "(null)".  If an error occurred during
  // the conversion, "(failed to convert from wide string)" is
  // returned.
  static String ShowWideCString(const wchar_t* wide_c_str);

  // Similar to ShowWideCString(), except that this function encloses
  // the converted string in double quotes.
  static String ShowWideCStringQuoted(const wchar_t* wide_c_str);

  // Compares two wide C strings.  Returns true iff they have the same
  // content.
  //
  // Unlike wcscmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool WideCStringEquals(const wchar_t* lhs, const wchar_t* rhs);

  // Compares two C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike strcasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CaseInsensitiveCStringEquals(const char* lhs,
                                           const char* rhs);

  // Compares two wide C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike wcscasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL wide C string,
  // including the empty string.
  // NB: The implementations on different platforms slightly differ.
  // On windows, this method uses _wcsicmp which compares according to LC_CTYPE
  // environment variable. On GNU platform this method uses wcscasecmp
  // which compares according to LC_CTYPE category of the current locale.
  // On MacOS X, it uses towlower, which also uses LC_CTYPE category of the
  // current locale.
  static bool CaseInsensitiveWideCStringEquals(const wchar_t* lhs,
                                               const wchar_t* rhs);

  // Formats a list of arguments to a String, using the same format
  // spec string as for printf.
  //
  // We do not use the StringPrintf class as it is not universally
  // available.
  //
  // The result is limited to 4096 characters (including the tailing
  // 0).  If 4096 characters are not enough to format the input,
  // "<buffer exceeded>" is returned.
  static String Format(const char* format, ...);

  // C'tors

  // The default c'tor constructs a NULL string.
  String() : c_str_(NULL), length_(0) {}

  // Constructs a String by cloning a 0-terminated C string.
  String(const char* a_c_str) {  // NOLINT
    if (a_c_str == NULL) {
      c_str_ = NULL;
      length_ = 0;
    } else {
      ConstructNonNull(a_c_str, strlen(a_c_str));
    }
  }

  // Constructs a String by copying a given number of chars from a
  // buffer.  E.g. String("hello", 3) creates the string "hel",
  // String("a\0bcd", 4) creates "a\0bc", String(NULL, 0) creates "",
  // and String(NULL, 1) results in access violation.
  String(const char* buffer, size_t a_length) {
    ConstructNonNull(buffer, a_length);
  }

  // The copy c'tor creates a new copy of the string.  The two
  // String objects do not share content.
  String(const String& str) : c_str_(NULL), length_(0) { *this = str; }

  // D'tor.  String is intended to be a final class, so the d'tor
  // doesn't need to be virtual.
  ~String() { delete[] c_str_; }

  // Allows a String to be implicitly converted to an ::std::string or
  // ::string, and vice versa.  Converting a String containing a NULL
  // pointer to ::std::string or ::string is undefined behavior.
  // Converting a ::std::string or ::string containing an embedded NUL
  // character to a String will result in the prefix up to the first
  // NUL character.
  String(const ::std::string& str) {
    ConstructNonNull(str.c_str(), str.length());
  }

  operator ::std::string() const { return ::std::string(c_str(), length()); }

#if GTEST_HAS_GLOBAL_STRING
  String(const ::string& str) {
    ConstructNonNull(str.c_str(), str.length());
  }

  operator ::string() const { return ::string(c_str(), length()); }
#endif  // GTEST_HAS_GLOBAL_STRING

  // Returns true iff this is an empty string (i.e. "").
  bool empty() const { return (c_str() != NULL) && (length() == 0); }

  // Compares this with another String.
  // Returns < 0 if this is less than rhs, 0 if this is equal to rhs, or > 0
  // if this is greater than rhs.
  int Compare(const String& rhs) const;

  // Returns true iff this String equals the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator==(const char* a_c_str) const { return Compare(a_c_str) == 0; }

  // Returns true iff this String is less than the given String.  A
  // NULL string is considered less than "".
  bool operator<(const String& rhs) const { return Compare(rhs) < 0; }

  // Returns true iff this String doesn't equal the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator!=(const char* a_c_str) const { return !(*this == a_c_str); }

  // Returns true iff this String ends with the given suffix.  *Any*
  // String is considered to end with a NULL or empty suffix.
  bool EndsWith(const char* suffix) const;

  // Returns true iff this String ends with the given suffix, not considering
  // case. Any String is considered to end with a NULL or empty suffix.
  bool EndsWithCaseInsensitive(const char* suffix) const;

  // Returns the length of the encapsulated string, or 0 if the
  // string is NULL.
  size_t length() const { return length_; }

  // Gets the 0-terminated C string this String object represents.
  // The String object still owns the string.  Therefore the caller
  // should NOT delete the return value.
  const char* c_str() const { return c_str_; }

  // Assigns a C string to this object.  Self-assignment works.
  const String& operator=(const char* a_c_str) {
    return *this = String(a_c_str);
  }

  // Assigns a String object to this object.  Self-assignment works.
  const String& operator=(const String& rhs) {
    if (this != &rhs) {
      delete[] c_str_;
      if (rhs.c_str() == NULL) {
        c_str_ = NULL;
        length_ = 0;
      } else {
        ConstructNonNull(rhs.c_str(), rhs.length());
      }
    }

    return *this;
  }

 private:
  // Constructs a non-NULL String from the given content.  This
  // function can only be called when c_str_ has not been allocated.
  // ConstructNonNull(NULL, 0) results in an empty string ("").
  // ConstructNonNull(NULL, non_zero) is undefined behavior.
  void ConstructNonNull(const char* buffer, size_t a_length) {
    char* const str = new char[a_length + 1];
    memcpy(str, buffer, a_length);
    str[a_length] = '\0';
    c_str_ = str;
    length_ = a_length;
  }

  const char* c_str_;
  size_t length_;
};  // class String

// Streams a String to an ostream.  Each '\0' character in the String
// is replaced with "\\0".
inline ::std::ostream& operator<<(::std::ostream& os, const String& str) {
  if (str.c_str() == NULL) {
    os << "(null)";
  } else {
    const char* const c_str = str.c_str();
    for (size_t i = 0; i != str.length(); i++) {
      if (c_str[i] == '\0') {
        os << "\\0";
      } else {
        os << c_str[i];
      }
    }
  }
  return os;
}

// Gets the content of the stringstream's buffer as a String.  Each '\0'
// character in the buffer is replaced with "\\0".
GTEST_API_ String StringStreamToString(::std::stringstream* stream);

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".

// Declared here but defined in gtest.h, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: keith.ray@gmail.com (Keith Ray)
//
// Google Test filepath utilities
//
// This header file declares classes and functions used internally by
// Google Test.  They are subject to change without notice.
//
// This file is #included in <gtest/internal/gtest-internal.h>.
// Do not include this header file separately!

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_


namespace testing {
namespace internal {

// FilePath - a class for file and directory pathname manipulation which
// handles platform-specific conventions (like the pathname separator).
// Used for helper functions for naming files in a directory for xml output.
// Except for Set methods, all methods are const or static, which provides an
// "immutable value object" -- useful for peace of mind.
// A FilePath with a value ending in a path separator ("like/this/") represents
// a directory, otherwise it is assumed to represent a file. In either case,
// it may or may not represent an actual file or directory in the file system.
// Names are NOT checked for syntax correctness -- no checking for illegal
// characters, malformed paths, etc.

class GTEST_API_ FilePath {
 public:
  FilePath() : pathname_("") { }
  FilePath(const FilePath& rhs) : pathname_(rhs.pathname_) { }

  explicit FilePath(const char* pathname) : pathname_(pathname) {
    Normalize();
  }

  explicit FilePath(const String& pathname) : pathname_(pathname) {
    Normalize();
  }

  FilePath& operator=(const FilePath& rhs) {
    Set(rhs);
    return *this;
  }

  void Set(const FilePath& rhs) {
    pathname_ = rhs.pathname_;
  }

  String ToString() const { return pathname_; }
  const char* c_str() const { return pathname_.c_str(); }

  // Returns the current working directory, or "" if unsuccessful.
  static FilePath GetCurrentDir();

  // Given directory = "dir", base_name = "test", number = 0,
  // extension = "xml", returns "dir/test.xml". If number is greater
  // than zero (e.g., 12), returns "dir/test_12.xml".
  // On Windows platform, uses \ as the separator rather than /.
  static FilePath MakeFileName(const FilePath& directory,
                               const FilePath& base_name,
                               int number,
                               const char* extension);

  // Given directory = "dir", relative_path = "test.xml",
  // returns "dir/test.xml".
  // On Windows, uses \ as the separator rather than /.
  static FilePath ConcatPaths(const FilePath& directory,
                              const FilePath& relative_path);

  // Returns a pathname for a file that does not currently exist. The pathname
  // will be directory/base_name.extension or
  // directory/base_name_<number>.extension if directory/base_name.extension
  // already exists. The number will be incremented until a pathname is found
  // that does not already exist.
  // Examples: 'dir/foo_test.xml' or 'dir/foo_test_1.xml'.
  // There could be a race condition if two or more processes are calling this
  // function at the same time -- they could both pick the same filename.
  static FilePath GenerateUniqueFileName(const FilePath& directory,
                                         const FilePath& base_name,
                                         const char* extension);

  // Returns true iff the path is NULL or "".
  bool IsEmpty() const { return c_str() == NULL || *c_str() == '\0'; }

  // If input name has a trailing separator character, removes it and returns
  // the name, otherwise return the name string unmodified.
  // On Windows platform, uses \ as the separator, other platforms use /.
  FilePath RemoveTrailingPathSeparator() const;

  // Returns a copy of the FilePath with the directory part removed.
  // Example: FilePath("path/to/file").RemoveDirectoryName() returns
  // FilePath("file"). If there is no directory part ("just_a_file"), it returns
  // the FilePath unmodified. If there is no file part ("just_a_dir/") it
  // returns an empty FilePath ("").
  // On Windows platform, '\' is the path separator, otherwise it is '/'.
  FilePath RemoveDirectoryName() const;

  // RemoveFileName returns the directory path with the filename removed.
  // Example: FilePath("path/to/file").RemoveFileName() returns "path/to/".
  // If the FilePath is "a_file" or "/a_file", RemoveFileName returns
  // FilePath("./") or, on Windows, FilePath(".\\"). If the filepath does
  // not have a file, like "just/a/dir/", it returns the FilePath unmodified.
  // On Windows platform, '\' is the path separator, otherwise it is '/'.
  FilePath RemoveFileName() const;

  // Returns a copy of the FilePath with the case-insensitive extension removed.
  // Example: FilePath("dir/file.exe").RemoveExtension("EXE") returns
  // FilePath("dir/file"). If a case-insensitive extension is not
  // found, returns a copy of the original FilePath.
  FilePath RemoveExtension(const char* extension) const;

  // Creates directories so that path exists. Returns true if successful or if
  // the directories already exist; returns false if unable to create
  // directories for any reason. Will also return false if the FilePath does
  // not represent a directory (that is, it doesn't end with a path separator).
  bool CreateDirectoriesRecursively() const;

  // Create the directory so that path exists. Returns true if successful or
  // if the directory already exists; returns false if unable to create the
  // directory for any reason, including if the parent directory does not
  // exist. Not named "CreateDirectory" because that's a macro on Windows.
  bool CreateFolder() const;

  // Returns true if FilePath describes something in the file-system,
  // either a file, directory, or whatever, and that something exists.
  bool FileOrDirectoryExists() const;

  // Returns true if pathname describes a directory in the file-system
  // that exists.
  bool DirectoryExists() const;

  // Returns true if FilePath ends with a path separator, which indicates that
  // it is intended to represent a directory. Returns false otherwise.
  // This does NOT check that a directory (or file) actually exists.
  bool IsDirectory() const;

  // Returns true if pathname describes a root directory. (Windows has one
  // root directory per disk drive.)
  bool IsRootDirectory() const;

  // Returns true if pathname describes an absolute path.
  bool IsAbsolutePath() const;

 private:
  // Replaces multiple consecutive separators with a single separator.
  // For example, "bar///foo" becomes "bar/foo". Does not eliminate other
  // redundancies that might be in a pathname involving "." or "..".
  //
  // A pathname with multiple consecutive separators may occur either through
  // user error or as a result of some scripts or APIs that generate a pathname
  // with a trailing separator. On other platforms the same API or script
  // may NOT generate a pathname with a trailing "/". Then elsewhere that
  // pathname may have another "/" and pathname components added to it,
  // without checking for the separator already being there.
  // The script language and operating system may allow paths like "foo//bar"
  // but some of the functions in FilePath will not handle that correctly. In
  // particular, RemoveTrailingPathSeparator() only removes one separator, and
  // it is called in CreateDirectoriesRecursively() assuming that it will change
  // a pathname from directory syntax (trailing separator) to filename syntax.
  //
  // On Windows this method also replaces the alternate path separator '/' with
  // the primary path separator '\\', so that for example "bar\\/\\foo" becomes
  // "bar\\foo".

  void Normalize();

  // Returns a pointer to the last occurence of a valid path separator in
  // the FilePath. On Windows, for example, both '/' and '\' are valid path
  // separators. Returns NULL if no path separator was found.
  const char* FindLastPathSeparator() const;

  String pathname_;
};  // class FilePath

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_
// This file was GENERATED by command:
//     pump.py gtest-type-util.h.pump
// DO NOT EDIT BY HAND!!!

// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Type utilities needed for implementing typed and type-parameterized
// tests.  This file is generated by a SCRIPT.  DO NOT EDIT BY HAND!
//
// Currently we support at most 50 types in a list, and at most 50
// type-parameterized tests in one type-parameterized test case.
// Please contact googletestframework@googlegroups.com if you need
// more.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_


// #ifdef __GNUC__ is too general here.  It is possible to use gcc without using
// libstdc++ (which is where cxxabi.h comes from).
# ifdef __GLIBCXX__
#  include <cxxabi.h>
# elif defined(__HP_aCC)
#  include <acxx_demangle.h>
# endif  // __GLIBCXX__

namespace testing {
namespace internal {

// GetTypeName<T>() returns a human-readable name of type T.
// NB: This function is also used in Google Mock, so don't move it inside of
// the typed-test-only section below.
template <typename T>
String GetTypeName() {
# if GTEST_HAS_RTTI

  const char* const name = typeid(T).name();
#  if defined(__GLIBCXX__) || defined(__HP_aCC)
  int status = 0;
  // gcc's implementation of typeid(T).name() mangles the type name,
  // so we have to demangle it.
#   ifdef __GLIBCXX__
  using abi::__cxa_demangle;
#   endif // __GLIBCXX__
  char* const readable_name = __cxa_demangle(name, 0, 0, &status);
  const String name_str(status == 0 ? readable_name : name);
  free(readable_name);
  return name_str;
#  else
  return name;
#  endif  // __GLIBCXX__ || __HP_aCC

# else

  return "<type>";

# endif  // GTEST_HAS_RTTI
}

#if GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

// AssertyTypeEq<T1, T2>::type is defined iff T1 and T2 are the same
// type.  This can be used as a compile-time assertion to ensure that
// two types are equal.

template <typename T1, typename T2>
struct AssertTypeEq;

template <typename T>
struct AssertTypeEq<T, T> {
  typedef bool type;
};

// A unique type used as the default value for the arguments of class
// template Types.  This allows us to simulate variadic templates
// (e.g. Types<int>, Type<int, double>, and etc), which C++ doesn't
// support directly.
struct None {};

// The following family of struct and struct templates are used to
// represent type lists.  In particular, TypesN<T1, T2, ..., TN>
// represents a type list with N types (T1, T2, ..., and TN) in it.
// Except for Types0, every struct in the family has two member types:
// Head for the first type in the list, and Tail for the rest of the
// list.

// The empty type list.
struct Types0 {};

// Type lists of length 1, 2, 3, and so on.

template <typename T1>
struct Types1 {
  typedef T1 Head;
  typedef Types0 Tail;
};
template <typename T1, typename T2>
struct Types2 {
  typedef T1 Head;
  typedef Types1<T2> Tail;
};

template <typename T1, typename T2, typename T3>
struct Types3 {
  typedef T1 Head;
  typedef Types2<T2, T3> Tail;
};

template <typename T1, typename T2, typename T3, typename T4>
struct Types4 {
  typedef T1 Head;
  typedef Types3<T2, T3, T4> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct Types5 {
  typedef T1 Head;
  typedef Types4<T2, T3, T4, T5> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
struct Types6 {
  typedef T1 Head;
  typedef Types5<T2, T3, T4, T5, T6> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
struct Types7 {
  typedef T1 Head;
  typedef Types6<T2, T3, T4, T5, T6, T7> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
struct Types8 {
  typedef T1 Head;
  typedef Types7<T2, T3, T4, T5, T6, T7, T8> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
struct Types9 {
  typedef T1 Head;
  typedef Types8<T2, T3, T4, T5, T6, T7, T8, T9> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
struct Types10 {
  typedef T1 Head;
  typedef Types9<T2, T3, T4, T5, T6, T7, T8, T9, T10> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11>
struct Types11 {
  typedef T1 Head;
  typedef Types10<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12>
struct Types12 {
  typedef T1 Head;
  typedef Types11<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13>
struct Types13 {
  typedef T1 Head;
  typedef Types12<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14>
struct Types14 {
  typedef T1 Head;
  typedef Types13<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15>
struct Types15 {
  typedef T1 Head;
  typedef Types14<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16>
struct Types16 {
  typedef T1 Head;
  typedef Types15<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17>
struct Types17 {
  typedef T1 Head;
  typedef Types16<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18>
struct Types18 {
  typedef T1 Head;
  typedef Types17<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19>
struct Types19 {
  typedef T1 Head;
  typedef Types18<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20>
struct Types20 {
  typedef T1 Head;
  typedef Types19<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21>
struct Types21 {
  typedef T1 Head;
  typedef Types20<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22>
struct Types22 {
  typedef T1 Head;
  typedef Types21<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23>
struct Types23 {
  typedef T1 Head;
  typedef Types22<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24>
struct Types24 {
  typedef T1 Head;
  typedef Types23<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25>
struct Types25 {
  typedef T1 Head;
  typedef Types24<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26>
struct Types26 {
  typedef T1 Head;
  typedef Types25<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27>
struct Types27 {
  typedef T1 Head;
  typedef Types26<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28>
struct Types28 {
  typedef T1 Head;
  typedef Types27<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29>
struct Types29 {
  typedef T1 Head;
  typedef Types28<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30>
struct Types30 {
  typedef T1 Head;
  typedef Types29<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31>
struct Types31 {
  typedef T1 Head;
  typedef Types30<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32>
struct Types32 {
  typedef T1 Head;
  typedef Types31<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33>
struct Types33 {
  typedef T1 Head;
  typedef Types32<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34>
struct Types34 {
  typedef T1 Head;
  typedef Types33<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35>
struct Types35 {
  typedef T1 Head;
  typedef Types34<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36>
struct Types36 {
  typedef T1 Head;
  typedef Types35<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37>
struct Types37 {
  typedef T1 Head;
  typedef Types36<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38>
struct Types38 {
  typedef T1 Head;
  typedef Types37<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39>
struct Types39 {
  typedef T1 Head;
  typedef Types38<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40>
struct Types40 {
  typedef T1 Head;
  typedef Types39<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41>
struct Types41 {
  typedef T1 Head;
  typedef Types40<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42>
struct Types42 {
  typedef T1 Head;
  typedef Types41<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43>
struct Types43 {
  typedef T1 Head;