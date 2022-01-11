#include "yours.hpp"
#include "given.hpp"


Mat yours::visualizeHoughSpace(vector< vector<Mat> >& houghSpace)
{
	Mat houghPlot(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, houghSpace.at(0).at(0).type());

	for (vector< vector<Mat> >::const_iterator it = houghSpace.begin(); it != houghSpace.end(); it++)
	{
		for (vector<Mat>::const_iterator img = (*it).begin(); img != (*it).end(); img++) 
		{
			for (int i = 0; i < (*img).rows; ++i) 
			{
				for (int j = 0; j < (*img).cols; ++j) 
				{
					houghPlot.at<float>(i, j) += (*img).at<float>(i, j);
				}
			}
		}
	}
	normalize(houghPlot, houghPlot, 0, 1, NORM_MINMAX);
	given::showImage(houghPlot, "Hough Space", 0);
	//imwrite("Hough_Space.png", houghPlot * 255);

	return houghPlot;
}


void yours::makeFFTObjectMask(vector<Mat>& templ, double scale, double angle, Mat& fftMask)
{
	Mat binaryEdge = templ[0];
	Mat complexGradients[2] , Magnitu;
	split(templ[1], complexGradients);
	binaryEdge = given::rotateAndScale(binaryEdge, angle, scale);
	complexGradients[0] = given::rotateAndScale(complexGradients[0], angle, scale);
	complexGradients[1] = given::rotateAndScale(complexGradients[1], angle, scale);

	Mat Magnitu_1, Delta;
	cartToPolar(complexGradients[0], complexGradients[1], Magnitu_1, Delta);
	Delta = Delta + angle;

	polarToCart(Magnitu_1, Delta, complexGradients[0], complexGradients[1]);
	float mag_sum = 0;
	for (int i = 0; i < Magnitu_1.rows; i++)
	{
		for (int j = 0; j < Magnitu_1.cols; j++) {
			mag_sum = mag_sum + Magnitu_1.at<float>(i, j);
		}
	}

	complexGradients[0] = complexGradients[0] / mag_sum;
	complexGradients[1] = complexGradients[1] / mag_sum;
	multiply(complexGradients[0], binaryEdge, complexGradients[0]);
	multiply(complexGradients[1], binaryEdge, complexGradients[1]);

	Mat newMask[] = { Mat::zeros(fftMask.rows, fftMask.cols, CV_32FC1),
					 Mat::zeros(fftMask.rows, fftMask.cols, CV_32FC1) };
	Mat complexMask;
	complexGradients[0].copyTo(newMask[0](cv::Rect(0, 0, complexGradients[0].cols, complexGradients[0].rows)));
	complexGradients[0].copyTo(newMask[1](cv::Rect(0, 0, complexGradients[1].cols, complexGradients[1].rows)));
	merge(newMask, 2, complexMask);
	given::circShift(complexMask, complexMask, -complexGradients[0].cols / 2, -complexGradients[0].rows / 2);
	dft(complexMask, fftMask, DFT_COMPLEX_OUTPUT);

}


vector<vector<Mat> > yours::generalHough(Mat& gradImage, vector<Mat>& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange)
{
	vector< vector<Mat> > Hough;
	double scaleStepSize = (scaleRange[1] - scaleRange[0]) / (scaleSteps-1);
	double angleStepSize = (angleRange[1] - angleRange[0]) / angleSteps;
	// new Definition

	Mat imageFFTMask(gradImage.rows, gradImage.cols, CV_32FC2);

	Mat objectFFTMask(gradImage.rows, gradImage.cols, CV_32FC2);

	dft(gradImage, imageFFTMask, DFT_COMPLEX_OUTPUT);
	// Transform

	for (int a_scale = 0; a_scale < (int)scaleSteps; ++a_scale)
	{

		double scale = scaleRange[0] + a_scale * scaleStepSize;
		vector<Mat> orientationHough;

		for (int a_orientation = 0; a_orientation < (int)angleSteps; ++a_orientation) 
		{

			double angle = angleRange[0] + a_orientation * angleStepSize;

			makeFFTObjectMask(templ, scale, angle, objectFFTMask);

			Mat correlation(imageFFTMask.rows, imageFFTMask.cols, imageFFTMask.type());

			// calculate correlation
			mulSpectrums(imageFFTMask, objectFFTMask, correlation, 0, true);
			//0 no flags
			//true conjugate objectFFTMask before multiplication

			idft(correlation, correlation);

			Mat Image(correlation.rows, correlation.cols, CV_32FC1);

			for (int a = 0; a < correlation.rows; ++a)
			{
				for (int b = 0; b < correlation.cols; ++b)
				{
					// cout << correlation.at<Vec2f>(i, j) << endl;
					Image.at<float>(a, b) = sqrt(pow(correlation.at<Vec2f>(a, b)[0], 2) + pow(correlation.at<Vec2f>(a, b)[1], 2));
				}
			}
			orientationHough.push_back(Image);
		}
		Hough.push_back(orientationHough);
	}
	return Hough;

}


Mat yours::binarizeGradientImage(Mat& src, double threshold)
{
	Mat dst = src.clone();
	Mat binary(dst.rows, dst.cols, CV_32FC1, Scalar(0));
	int edge_value = 0;
	double amplitude = 0;
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			amplitude = sqrt(dst.at<cv::Vec2f>(i, j)[0] * dst.at<cv::Vec2f>(i, j)[0] + dst.at<cv::Vec2f>(i, j)[1] * dst.at<cv::Vec2f>(i, j)[1]);

			if (edge_value < amplitude)
				edge_value = amplitude;
		}
	}
	for (int m = 0; m < dst.rows; m++) {
		for (int n = 0; n < dst.cols; n++) {
			amplitude = sqrt(dst.at<cv::Vec2f>(m, n)[0] * dst.at<cv::Vec2f>(m, n)[0] + dst.at<cv::Vec2f>(m, n)[1] * dst.at<cv::Vec2f>(m, n)[1]);
			if (amplitude > threshold* edge_value) {
				binary.at<float>(m, n) = 255;
			}
			else {
				binary.at<float>(m, n) = 0;
			}
		}
	}

	return binary;

}
