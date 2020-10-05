//g++-7 gaussian_blur.cpp -o gaussian_blur `pkg-config --cflags --libs opencv` -lstdc++fs -std=c++17

#include <opencv2\opencv.hpp>
#include <iostream>
#include <experimental\filesystem>

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv) {
	int num_pieces = 3;
	int num_transformations = 11;
	int num_backgrounds = 100-8+1;

	cv::Size kernel_size(7, 7);

	std::string base_dir = "E:\\Alonso\\JUPYTER_NOTEBOOK_OPENGL_v10_synth_blur_variety_backs_7x7";
	std::string dir_images = "new_images";
	std::string dir_blurred_images = "blurred_images";

	fs::create_directory(base_dir + "\\" + dir_blurred_images);

	for (int i = 0; i < num_pieces; ++i) {
		fs::create_directory(base_dir + "\\" + dir_blurred_images + "\\pieza" + std::to_string(i));
		for (int j = 0; j < num_transformations; ++j) {
			for (int k = 0; k < num_backgrounds; ++k) {
				cv::Mat image = cv::imread(base_dir + "\\" + dir_images + "\\pieza" + std::to_string(i) + "\\img" + std::to_string(j) + "_background" + std::to_string(k) + "_w.png");
				if (image.empty()) {
					std::cout << "name image = " << dir_images << "\\pieza" << i << "\\img" << j << "_background" << k << "_w.png\n";
					std::cout << "Could not open or find the image\n";
					std::cin.get(); //wait for any key press
					return -1;
				}
				cv::Mat image_blurred;
				cv::GaussianBlur(image, image_blurred, kernel_size, 0);
				cv::imwrite(base_dir + "\\" + dir_blurred_images + "\\pieza" + std::to_string(i) + "\\img" + std::to_string(j) + "_background" + std::to_string(k) + ".png", image_blurred);
			}
		}
	}
	return 0;
}