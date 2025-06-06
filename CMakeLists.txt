cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
project(nmpde_Wave_Equation_Project LANGUAGES CXX)

# Forza Boost corretto
set(BOOST_ROOT "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0")
set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
include_directories(BEFORE SYSTEM ${BOOST_INCLUDEDIR})

# Trova deal.II
find_package(deal.II REQUIRED
             HINTS ${DEAL_II_DIR} /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1)

add_executable(WaveFunction src/WaveFunction.cpp src/main.cpp)
deal_ii_setup_target(WaveFunction)
