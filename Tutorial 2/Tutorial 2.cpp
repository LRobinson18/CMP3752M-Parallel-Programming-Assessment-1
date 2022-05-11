#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

/*
Lewis Robinson - ROB15611294
CMP3752M - Parallel Programming Assessment 1
Code adapted from: https://github.com/wing8/OpenCL-Tutorials
Using Tutorial 2.cpp as a base for this parallel programming assessment
The kernel to display an image from Tutorial 2 and the hist_simple/scan_add_atomic kernels from Tutorial 3 were used as templates.
The histogram was used to determine the distribution of pixel values in the image, which began with a low contrast level.
It employs global memory and atomic operators, which are slower than other approaches, but the execution times are still rapid because the dataset is short.
The next stage was to make a cumulative histogram of the image, which shows the total number of pixels against the pixel values.
The cumulative histogram was then normalised to build a lookup table (LUT) of new values for the pixels in order to boost the image's contrast. 
This was accomplished by developing a kernel that multiplied each value in the cumulative histogram by 255/total pixels. As a consequence, the values ranged from 0-255.
The final kernel simply sets the values in the LUT for each pixel in the input picture.
Each kernel's execution time and memory transfer are likewise tracked and presented alongside the histogram.
The output image of this application has a greater, more balanced contrast than the input image. Greyscale (.pgm) and colour (.bmp) pictures of various sizes are supported by the code.
*/

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm"; // Images include brayford.bmp(colour), test,pgm(greyscale), test_large.pgm(greyscale)

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	// Part 2 - get user input for the bin number
	string userCommand;
	int bin_num = 0;
	std::cout << "Enter a bin number in range 0-256" << "\n";
	while (true) // while the user hasn't entered a valid number the program will keep running
	{
		getline(std::cin, userCommand); // gets input from user
		if (userCommand == "") { std::cout << "Please enter a number." << "\n"; continue; } // checks user input isn't empty

		try { bin_num = std::stoi(userCommand); } // attempt to convert the user input to an integer
		catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

		if (bin_num >= 0 && bin_num <= 256) { break; } // checks user input is in range
		else { std::cout << "Please enter a number in range 0-256." << "\n"; continue; }
	}

	// detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		// a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		// Part 3 - host operations
		// Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Part 4 - memory allocation
		// Set histogram bin size based on user input from earlier
		typedef int mytype;
		std::vector<mytype> H(bin_num);
		size_t hist_size = H.size() * sizeof(mytype);

		// device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_histogram_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_cumulative_histogram_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_LUT_output(context, CL_MEM_READ_WRITE, hist_size);

		// Part 5 - device operations
		// Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		// Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "img_old");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);

		// Create frequency histogram all pixel values (0-255) in the picture
		cl::Kernel kernel_hist_simple = cl::Kernel(program, "hist_simple");
		kernel_hist_simple.setArg(0, dev_image_input);
		kernel_hist_simple.setArg(1, dev_histogram_output);

		cl::Event hist_event; // Create event for the first histogram

		queue.enqueueNDRangeKernel(kernel_hist_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_event);
		queue.enqueueReadBuffer(dev_histogram_output, CL_TRUE, 0, hist_size, &H[0]);

		std::vector<mytype> CH(bin_num);

		queue.enqueueFillBuffer(dev_cumulative_histogram_output, 0, 0, hist_size);

		// Calculate cumulative histogram of all pixels in the picture
		cl::Kernel kernel_hist_cum = cl::Kernel(program, "hist_cum");
		kernel_hist_cum.setArg(0, dev_histogram_output);
		kernel_hist_cum.setArg(1, dev_cumulative_histogram_output);

		cl::Event cum_hist_event; // Create event for the cumulative histogram

		queue.enqueueNDRangeKernel(kernel_hist_cum, cl::NullRange, cl::NDRange(hist_size), cl::NullRange, NULL, &cum_hist_event);
		queue.enqueueReadBuffer(dev_cumulative_histogram_output, CL_TRUE, 0, hist_size, &CH[0]);

		std::vector<mytype> LUT(bin_num);

		queue.enqueueFillBuffer(dev_LUT_output, 0, 0, hist_size);

		// The LUT normalises the cumulative histogram, decreasing the value of the pixels to increase the contrast
		cl::Kernel kernel_LUT = cl::Kernel(program, "hist_lut");
		kernel_LUT.setArg(0, dev_cumulative_histogram_output);
		kernel_LUT.setArg(1, dev_LUT_output);

		cl::Event lut_event; // Create event for the Look-up table (LUT)

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(hist_size), cl::NullRange, NULL, &lut_event);
		queue.enqueueReadBuffer(dev_LUT_output, CL_TRUE, 0, hist_size, &LUT[0]);

		// Assign the new pixel values from the lookup table to the new image
		cl::Kernel kernel_BackProj = cl::Kernel(program, "back_proj");
		kernel_BackProj.setArg(0, dev_image_input);
		kernel_BackProj.setArg(1, dev_LUT_output);
		kernel_BackProj.setArg(2, dev_image_output);

		cl::Event back_proj_event; // Create event for the projection of the new image

		// Print all of the histogram values and calculated kernel execution times/memory transfer
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(kernel_BackProj, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &back_proj_event);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		std::cout << std::endl;
		std::cout << "Histogram: " << H << std::endl;
		std::cout << "Histogram kernel execution time [ns]: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Histogram memory transfer: " << GetFullProfilingInfo(hist_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		std::cout << "Cumulative Histogram: " << CH << std::endl;
		std::cout << "Cumulative Histogram kernel execution time [ns]: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cum_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Cumulative Histogram memory transfer: " << GetFullProfilingInfo(cum_hist_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		std::cout << "Look-up table (LUT): " << LUT << std::endl;
		std::cout << "LUT kernel execution time [ns]: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lut_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "LUT memory transfer: " << GetFullProfilingInfo(lut_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		std::cout << "Vector kernel execution time [ns]: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - back_proj_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Vector memory transfer: " << GetFullProfilingInfo(back_proj_event, ProfilingResolution::PROF_US) << std::endl;
		
		// Final output image
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
