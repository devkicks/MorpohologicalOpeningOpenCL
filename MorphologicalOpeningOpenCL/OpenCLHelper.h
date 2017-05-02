#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "CL/cl.hpp"

// Function for creating an OpenCL program using .cl file as input
cl::Program CreateProgram(const std::string &file, int deviceType = CL_DEVICE_TYPE_CPU);

// Function for debugging the error in OpenCL program compilation or kernel execution
// Help from : http://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* getErrorString(cl_int error);
void processError(cl_int erro, std::string errorMessager);