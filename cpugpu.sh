rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_CXX_COMPILER=icpx
make CPUGPU && ./CPUGPU/CPUGPU

cd ..

rm -rf build-acpp
mkdir build-acpp && cd build-acpp

cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUSE_ACPP=ON -DACPP_TARGETS="omp" -DCMAKE_BUILD_TYPE=Release
make CPUGPU && ./CPUGPU/CPUGPU

cd ..
