cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(EM-torch-extension)

find_package(Torch REQUIRED)

add_subdirectory(pybind11)
add_executable(em-torch-ext em_torch_ext/TorchExtension.cpp em_torch_ext/TorchExtensionKernel.cu)
target_link_libraries(em-torch-ext "${TORCH_LIBRARIES}")
set_property(TARGET em-torch-ext PROPERTY CXX_STANDARD 14)