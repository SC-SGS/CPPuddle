# Using the scripts for building dependencies on Piz Daint

This project can be used without these scripts. These scripts merely help
building the dependencies with appropriate options.

`./build_dependencies.sh` will build HPX and Kokkos with CUDA and appropriate
options enabled. `./build_kokkos.sh` and `./build_hpx.sh` can be used to build
each project individually (e.g. if one has been updated).
`./clean_dependencies.sh` will delete the build directories. Running this can
be useful if e.g. the compiler or significant CMake options have changed which
requires a clean build.

Once the dependencies are built `./configure_build_directory.sh` will configure
a build directory with the correct options in
`repo_root/build/${CMAKE_BUILD_TYPE}`.

To actually build the project you need to manually go to the build directory.
To have the correct environment, first source `daint_env.sh`.

The build type for all builds is set in `daint_env.sh`. It is set to `Debug` by
default.
