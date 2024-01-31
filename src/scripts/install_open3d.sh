sudo apt update && apt install -y \
    git \
    cmake \
    wget \
    libstdc++6 \
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    clang \
    libc++-dev \
    libc++abi-dev \
    libsdl2-dev \
    ninja-build \
    libxi-dev \
    libtbb-dev \
    libosmesa6-dev \
    libudev-dev \
    autoconf \
    libtool \
    git \
    cmake

#Install open3d
git clone https://github.com/isl-org/Open3D.git
cd Open3D
git reset --hard ff22900
mkdir build
cd build
cmake -DENABLE_HEADLESS_RENDERING=ON \
                -DBUILD_GUI=OFF \
                -DUSE_SYSTEM_GLEW=OFF \
                -DUSE_SYSTEM_GLFW=OFF \
                ..
make -j8
sudo make install-pip-package