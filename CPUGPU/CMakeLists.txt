add_executable(CPUGPU main.cpp)

if(NOT USE_ACPP)
target_link_libraries(CPUGPU PUBLIC MKL::MKL)
endif()

add_sycl_to_target(TARGET CPUGPU SOURCES main.cpp)
