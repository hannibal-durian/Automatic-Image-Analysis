//
// Implement the required functions here.
//
#include "yours.hpp"
#include "given.hpp"


void yours::preprocessImage(cv::Mat& src, cv::Mat& dst, int bin_thresh, int n_erosions) {


    cv::Mat tmp = src;
    if(src.channels() > 1)
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);

   // get binary image (white foreground, black background)
	cv::Mat bin;
	cv::threshold(tmp, bin, bin_thresh, 255, cv::THRESH_BINARY_INV);


    // use erosion to get rid of small objects and break connections between leafs
    // use a 3x3 structuring element (cv::Mat::ones(3, 3, CV_8UC1))
	cv::Mat element = cv::Mat::ones(3, 3, CV_8UC1);
	cv::erode(bin, dst, element, cv::Point(-1, -1), n_erosions);

}

cv::Mat yours::getFourierDescriptor(const cv::Mat& contour) {
    // convert the contour to an adequate type and make the discrete fourier transform
    // use OpenCVs implementation of the DFT
	cv::Mat Con_Float,dft;
	contour.convertTo(Con_Float, CV_32F);
	cv::dft(Con_Float, dft);
	return dft;

}

cv::Mat yours::normalizeFourierDescriptor(const cv::Mat& fd, int n) {

	//given::plotFourierDescriptor(fd, "fd not normalized", 0);
    // translation invariance F(0) = 0
	cv::Mat fd_trans = fd.clone();
	fd_trans.at<cv::Vec2f>(0, 0)[0] = 0;
	fd_trans.at<cv::Vec2f>(0, 0)[1] = 0;
	//given::plotFourierDescriptor(fd_trans, "fd translation invariant", 0);


    // scale invariance F(i) = F(i)/|F(1)|
    // What if |F(1)| = 0?
	cv::Mat fd_scale = fd_trans.clone();
	float fd_norm = sqrt(fd_scale.at<cv::Vec2f>(1, 0)[0] * fd_scale.at<cv::Vec2f>(1, 0)[0] + fd_scale.at<cv::Vec2f>(1, 0)[1] * fd_scale.at<cv::Vec2f>(1, 0)[1]);
	int i = 1;
	while (fd_norm == 0) {
		fd_norm = sqrt(fd_scale.at<cv::Vec2f>(i, 0)[0] * fd_scale.at<cv::Vec2f>(i, 0)[0] + fd_scale.at<cv::Vec2f>(i, 0)[1] * fd_scale.at<cv::Vec2f>(i, 0)[1]);
		i++;
	}

	for (int i = 0; i <= fd_scale.rows - 1; i++) {

		fd_scale.row(i) = fd_scale.row(i) / fd_norm;

	}

	//given::plotFourierDescriptor(fd_scale, "fd translation and scale invariant", 0);


    // rotation invariance F = |F|
    // There are some useful OpenCV functions such as cartToPolar
	cv::Mat fd_ro = fd_scale.clone();
	cv::Mat fd_ro_angle = cv::Mat::zeros(n, 1, CV_32FC1);

	std::vector<cv::Mat> channels;
	split(fd_ro, channels);

	cartToPolar(channels.at(0), channels.at(1), channels.at(0), fd_ro_angle);
	channels.at(1) = 0;

	merge(channels, fd_ro);

	//given::plotFourierDescriptor(fd_ro, "fd translation, scale, and rotation invariant", 0);


    // smaller sensitivity for details
    // This one is a bit tricky. How does your descriptor look like?
    // Where are the high frequencies and where are the negative indices?
	cv::Mat fd_noise = cv::Mat::zeros(n, 1, CV_32FC2);
	for (int i = 0; i < n; i++) {
		if (i < n / 2) {
			fd_noise.at<cv::Vec2f>(i, 0)[0] = fd_ro.at<cv::Vec2f>(i, 0)[0];
			fd_noise.at<cv::Vec2f>(i, 0)[1] = fd_ro.at<cv::Vec2f>(i, 0)[1];
		}
		else {
			fd_noise.at<cv::Vec2f>(i, 0)[0] = fd_ro.at<cv::Vec2f>(fd_ro.rows - n + i, 0)[0];
			fd_noise.at<cv::Vec2f>(i, 0)[1] = fd_ro.at<cv::Vec2f>(fd_ro.rows - n + i, 0)[1];
		}
	}

	//given::plotFourierDescriptor(fd_noise, "fd translation, scale, and rotation invariant, smaller sensitivity", 0);

	cv::Mat fd_nor = cv::Mat::zeros(n, 1, CV_32FC1);
	for (int i = 0; i < n; i++)
		fd_nor.at<cv::Vec2f>(i, 0)[0] = fd_noise.at<cv::Vec2f>(i, 0)[0];
	
	cv::normalize(fd_nor, fd_nor, 0, 1, cv::NORM_MINMAX);
	return fd_nor;
}

int yours::classifyFourierDescriptor(const cv::Mat& fd, const std::vector<cv::Mat>& class_templates, float thresh) {
    // loop over templates and find closest, return index
    // use cv::norm as a disctance metric
	std::vector<double>norm_error;
	int index = 0;
	int i = 0;
	float errm = thresh;
	for (auto const& a : class_templates) {
		float error = cv::norm(fd, a);
		if (error < errm) {
			errm = error;
			index = i;
		}
		i += 1;
	}
	if (errm < thresh) {
		return index;
	}
	else {
		return -1;
	}
		
	


	//for (int i = 0; i < norm_error.size(); i++) {
	//	if (min_error > norm_error[i]) {
	//		min_error > norm_error[i];
	//		index = i;
	//	}
	//}
	//if (min_error > thresh) {
	//	return -1;
	//}
	//else
	//	return index;

	}
	

