#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <fstream>
#include "opencv2\opencv.hpp"
#include "CL\cl.hpp"
#include "OpenCLHelper.h"


int main(int argc, char ** argv) {

	// I was unable to convert the image into raw images using imagemagick
	// it gives me the following error, which I did not have time to debug
	//
	// >convert testImage2.jpg -depth 0 gray:testImage.raw
	//	Invalid Parameter - -depth

	// I am using OpenCV just to load and convert data into relevant format


	// Load an input image

	// The OpenCV binaries I had have been compiled using Visual Studio 2010 - Therefore this project might ask for additional dll libraries if VS 2010 is not installed
	cv::Mat inImage = cv::imread("Images/testImage3.png");
	if (inImage.empty())
	{
		std::cout << "Unable to load the image" << std::endl;
		exit(EXIT_FAILURE);
	}
	// show the image in a window
	cv::imshow("inImage", inImage);
	cv::Mat cInImage;
	cv::cvtColor(inImage, cInImage, CV_RGB2GRAY);
	
	if (cInImage.cols * cInImage.rows % 64 != 0)
	{
		std::cout << "Invalid size - please provide image with width*height multiple of 64" << std::endl;
		exit(EXIT_FAILURE);
	}

	// I did not had time to implement the part where multiple erosions can take place.
	// My idea is to have an output image buffer, that I can feed to multipl kernels.
	// Unfortunately I was not able make the enqueueReadImage function work - as I do not understand well the parameters it needs
	// Therefore, my implementation uses a buffer as an output - 

	// Future improvements will be to use ImageBuffer - write_image function in the kernel
	//												  - Connecting multiple kernels so that we can execute one kernel after another without downloading data back to host.

	int numErosions = 1;
	// A single kernel is used for both operations
	cl::Program program = CreateProgram("morphological_operations.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices.front();

	// Setup image as input
	cl::Image2D clImage = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_UNORM_INT8), cInImage.cols, cInImage.rows, 0, (void*)cInImage.data);
	cl::Buffer clResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*cInImage.cols*cInImage.rows);
	
	// Set up kernel for Erosion
	cl::Kernel morphologicalErosion = cl::Kernel(program, "morphological_operations");
	morphologicalErosion.setArg(0, clImage);
	morphologicalErosion.setArg(1, clResult);
	morphologicalErosion.setArg(2, 1);/* 1 for erosion*/

	// Queue command for erosion Kernel
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(morphologicalErosion, cl::NullRange, cl::NDRange(cInImage.cols, cInImage.rows), cl::NullRange);

	// Transfer image back to host
	float* data = new float[cInImage.cols*cInImage.rows];
	queue.enqueueReadBuffer(clResult, CL_TRUE, 0, sizeof(float)*cInImage.cols*cInImage.rows, data);

	// prepare data to go to second kernel
	cl::Image2D clImage2 = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT), cInImage.cols, cInImage.rows, 0, (void*)data);
	cl::Buffer clResult2 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*cInImage.cols*cInImage.rows);

	// Set up kernel for Dilation
	cl::Kernel morphologicalDilation = cl::Kernel(program, "morphological_operations");
	morphologicalDilation.setArg(0, clImage2);
	morphologicalDilation.setArg(1, clResult2);
	morphologicalDilation.setArg(2, 0); /* 0 for dilation*/

	// Queue command for dilation and image read
	queue.enqueueNDRangeKernel(morphologicalDilation, cl::NullRange, cl::NDRange(cInImage.cols, cInImage.rows), cl::NullRange);
	queue.enqueueReadBuffer(clResult2, CL_TRUE, 0, sizeof(float)*cInImage.cols*cInImage.rows, data);

	// read image and convert to uchar to display as output - alternatively the image can be saved
	cv::Mat dispImage(inImage.rows, inImage.cols, CV_32FC1);
	dispImage.data = (uchar*)data;

	cv::Mat dImage;
	dispImage.convertTo(dImage, CV_8UC1);

	cv::imshow("MorphologicalOutput", dispImage);
	cv::waitKey(0);
}


