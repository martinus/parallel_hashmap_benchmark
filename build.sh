clang++ --std=c++17 \
    -L /home/martinus/git/Microsoft__vcpkg/packages/tbb_x64-linux/lib \
    -I /home/martinus/git/Microsoft__vcpkg/packages/tbb_x64-linux/include \
    -I $HOME/git/boostorg__unordered/include \
    -I $HOME/dev/boost_1_81_0/ \
    -DNDEBUG \
    -lboost_thread \
    -O3 -march=native \
    random_frequencies.cpp \
    -ltbb \
    -o bench
