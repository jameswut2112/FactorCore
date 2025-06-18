@echo off
set BUILD_DIR=build
set VCPKG_ROOT=D:\vcpkg
set PYTHON_ROOT=C:\Anaconda3\envs\py3

cmake -B %BUILD_DIR% -S . ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DPython_ROOT_DIR="%PYTHON_ROOT%"

cmake --build %BUILD_DIR% --config Release
