//
// Do not change this file.
//
#ifndef YOURS_H
#define YOURS_H

#include <iostream>
#include <opencv2/opencv.hpp>

namespace yours {

// Binarization and erosion of input image.
// src: Input image
// dst: Output image
// thresh: Binarization threshold
// num_erosions: Number on erosions.
void preprocessImage(cv::Mat& src, cv::Mat& dst, int bin_thresh, int n_erosions);

// Calculates the (unnormalized!) fourier descriptor from a list of points
// contour: 1xN 2-channel matrix, containing N points (x in first, y in second channel)
// returns: Fourier descriptor (not normalized)
cv::Mat getFourierDescriptor(const cv::Mat& contour);

// Normalize a given fourier descriptor
// fd: The given fourier descriptor as returned by getFourierDescriptor.
// descriptor_length: Number of used frequencies (should be even)
// returns: The normalized fourier descriptor
cv::Mat normalizeFourierDescriptor(const cv::Mat& fd, int descriptor_length);

// Classification of object by Fourier descriptor. Returns the index of the class that
// has the highest similarity with the given descriptor. Returns -1 if no distance
// value is above given threshold.
// Uses euclidean distance as distance metric. The euclidean distance is normilzed by descriptor dimension.
// fd: Fourier descriptor to classify.
// class_templates: A vector of reference Fourier descriptors representing the classses.
// thresh: Maximum distance value.
// returns: Index of class template with highest similarity/lowest distance or -1.
int classifyFourierDescriptor(const cv::Mat& fd, const std::vector<cv::Mat>& class_templates, float thresh);


}

#endif // YOURS_H
