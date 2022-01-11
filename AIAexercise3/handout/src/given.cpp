//============================================================================
// Author      : Ronny Haensch
// Version     : 1.1
// Copyright   : -
// Description : Given code for GHT exercise
//============================================================================
#include "given.hpp"


void given::showImage(const cv::Mat& img, std::string win, double dur) {
    // use copy for normalization
    cv::Mat copy = img.clone();
    if (img.channels() == 1) cv::normalize(img, copy, 0, 255, cv::NORM_MINMAX);

    copy.convertTo(copy, CV_8U);
    // create window and display image
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::imshow(win, copy);
    // wait
    if (dur >= 0) cv::waitKey(dur);
}


Mat given::calcDirectionalGrad(Mat& image, double sigma){

  // compute kernel size
  int ksize = max(sigma*3,3.);
  if (ksize % 2 == 0)  ksize++;
  double mu = ksize/2.0;

  // generate kernels for x- and y-direction
  double val, sum=0;
  Mat kernel(ksize, ksize, CV_32FC1);
  //Mat kernel_y(ksize, ksize, CV_32FC1);
  for(int i=0; i<ksize; i++){
      for(int j=0; j<ksize; j++){
        val  = pow((i+0.5-mu)/sigma,2);
        val += pow((j+0.5-mu)/sigma,2);
        val = exp(-0.5*val);
        sum += val;
        kernel.at<float>(i, j) = -(j+0.5-mu)*val;
     }
  }
  kernel /= sum;
  // use those kernels to compute gradient in x- and y-direction independently
  vector<Mat> grad(2);
  filter2D(image, grad[0], -1, kernel);
  filter2D(image, grad[1], -1, kernel.t());
  // combine both real-valued gradient images to a single complex-valued image
  Mat output;
  merge(grad, output);
  
  return output; 
}


Mat given::rotateAndScale(Mat& image, double angle, double scale){
    
    // create transformation matrices
    // translation to origin
    Mat T = Mat::eye(3, 3, CV_32FC1);    
    T.at<float>(0, 2) = -image.cols/2.0;
    T.at<float>(1, 2) = -image.rows/2.0;
    // rotation
    Mat R = Mat::eye(3, 3, CV_32FC1);
    R.at<float>(0, 0) =  cos(angle);
    R.at<float>(0, 1) = -sin(angle);
    R.at<float>(1, 0) =  sin(angle);
    R.at<float>(1, 1) =  cos(angle);
    // scale
    Mat S = Mat::eye(3, 3, CV_32FC1);    
    S.at<float>(0, 0) = scale;
    S.at<float>(1, 1) = scale;
    // combine
    Mat H = R*S*T;

    // compute corners of warped image
    Mat corners(1, 4, CV_32FC2);
    corners.at<Vec2f>(0, 0) = Vec2f(0,0);
    corners.at<Vec2f>(0, 1) = Vec2f(0,image.rows);
    corners.at<Vec2f>(0, 2) = Vec2f(image.cols,0);
    corners.at<Vec2f>(0, 3) = Vec2f(image.cols,image.rows);
    perspectiveTransform(corners, corners, H);
    
    // compute size of resulting image and allocate memory
    float x_start = min( min( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), min( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float x_end   = max( max( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), max( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float y_start = min( min( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), min( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
    float y_end   = max( max( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), max( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
       
    // create translation matrix in order to copy new object to image center
    T.at<float>(0, 0) = 1;
    T.at<float>(1, 1) = 1;
    T.at<float>(2, 2) = 1;
    T.at<float>(0, 2) = (x_end - x_start + 1)/2.0;
    T.at<float>(1, 2) = (y_end - y_start + 1)/2.0;
    
    // change homography to take necessary translation into account
    H = T * H;
    // warp image and copy it to output image
    Mat output;
    warpPerspective(image, output, H, Size(x_end - x_start + 1, y_end - y_start + 1), INTER_LINEAR);

    return output;
  
}


Mat given::visualizeResult(Mat& templateImage, Mat& queryImage, vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange){

    // some matrices to deal with color
    Mat red = queryImage.clone();
    Mat green = queryImage.clone();
    Mat blue = queryImage.clone();
    Mat tmp = Mat::zeros(queryImage.rows, queryImage.cols, CV_32FC1);
    
    // scale and angle of current object
    double scale, angle;
    
    // for all objects
    for(vector<Scalar>::iterator it = objList.begin(); it != objList.end(); it++){
        // compute scale and angle of current object
        scale = (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
        angle = ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0]);

        // use scale and angle in order to generate new binary mask of template
        Mat binMask = rotateAndScale(templateImage, angle, scale);

        // perform boundary checks
        Rect binArea = Rect(0, 0, binMask.cols, binMask.rows);
        Rect imgArea = Rect((*it).val[2]-binMask.cols/2., (*it).val[3]-binMask.rows/2, binMask.cols, binMask.rows);
        if ( (*it).val[2]-binMask.cols/2 < 0 ){
            binArea.x = abs( (*it).val[2]-binMask.cols/2 );
            binArea.width = binMask.cols - binArea.x;
            imgArea.x = 0;
            imgArea.width = binArea.width;
        }
        if ( (*it).val[3]-binMask.rows/2 < 0 ){
            binArea.y = abs( (*it).val[3]-binMask.rows/2 );
            binArea.height = binMask.rows - binArea.y;
            imgArea.y = 0;
            imgArea.height = binArea.height;
        }
        if ( (*it).val[2]-binMask.cols/2 + binMask.cols >= tmp.cols ){
            binArea.width = binMask.cols - ( (*it).val[2]-binMask.cols/2 + binMask.cols - tmp.cols );
            imgArea.width = binArea.width;
        }
        if ( (*it).val[3]-binMask.rows/2 + binMask.rows >= tmp.rows ){
            binArea.height = binMask.rows - ( (*it).val[3]-binMask.rows/2 + binMask.rows - tmp.rows );
            imgArea.height = binArea.height;
        }
        // copy this object instance in new image of correct size
        tmp.setTo(0);
        Mat binRoi = Mat(binMask, binArea);
        Mat imgRoi = Mat(tmp, imgArea);
        binRoi.copyTo(imgRoi);

        // delete found object from original image in order to reset pixel values with red (which are white up until now)
        binMask = 1 - binMask;
        imgRoi = Mat(red, imgArea);
        multiply(imgRoi, binRoi, imgRoi);
        imgRoi = Mat(green, imgArea);
        multiply(imgRoi, binRoi, imgRoi);
        imgRoi = Mat(blue, imgArea);
        multiply(imgRoi, binRoi, imgRoi);

        // change red channel
        red = red + tmp*255;
    }
    // generate color image
    vector<Mat> color;
    color.push_back(blue);
    color.push_back(green);
    color.push_back(red);
    Mat display;
    merge(color, display);
    return display;
}


void given::findHoughMaxima(vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList){

    // get maxima over scales and angles
    Mat maxImage = Mat::zeros(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1 );
    for(vector< vector<Mat> >::iterator it = houghSpace.begin(); it != houghSpace.end(); it++){
        for(vector<Mat>::iterator img = (*it).begin(); img != (*it).end(); img++){
            max(*img, maxImage, maxImage);
        }
    }
    // get global maxima
    double min, max;
    minMaxLoc(maxImage, &min, &max);

    // define threshold
    double threshold = objThresh * max;

    // spatial non-maxima suppression
    Mat bin = Mat(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1, -1);
    for(int y=0; y<maxImage.rows; y++){
        for(int x=0; x<maxImage.cols; x++){
            // init
            bool localMax = true;
            // check neighbors
            for(int i=-1; i<=1; i++){
                int new_y = y + i;
                if ((new_y < 0) or (new_y >= maxImage.rows)){
                    continue;
                }
                for(int j=-1; j<=1; j++){
                    int new_x = x + j;
                    if ((new_x < 0) or (new_x >= maxImage.cols)){
                    continue;
                    }
                    if (maxImage.at<float>(new_y, new_x) > maxImage.at<float>(y, x)){
                    localMax = false;
                    break;
                    }
                }
                if (!localMax)
                    break;
            }
            // check if local max is larger than threshold
            if ( (localMax) and (maxImage.at<float>(y, x) > threshold) ){
                bin.at<float>(y, x) = maxImage.at<float>(y, x);
            }
        }
    }
    
    // loop through hough space after non-max suppression and add objects to object list
    double scale, angle;
    scale = 0;
    for(vector< vector<Mat> >::iterator it = houghSpace.begin(); it != houghSpace.end(); it++, scale++){
        angle = 0;
        for(vector<Mat>::iterator img = (*it).begin(); img != (*it).end(); img++, angle++){
            for(int y=0; y<bin.rows; y++){
                for(int x=0; x<bin.cols; x++){
                    if ( (*img).at<float>(y, x) == bin.at<float>(y, x) ){
                    // create object list entry consisting of scale, angle, and position where object was detected
                    Scalar cur;
                    cur.val[0] = scale;
                    cur.val[1] = angle;
                    cur.val[2] = x;
                    cur.val[3] = y;
                    objList.push_back(cur);
                    }
                }
            }
        }
    }   
}


void given::circShift(Mat& in, Mat& out, int dx, int dy){

    Mat tmp = Mat::zeros(in.rows, in.cols, in.type());
 
    int x, y, new_x, new_y;

    for(y=0; y<in.rows; y++){

          // calulate new y-coordinate
          new_y = y + dy;
          if (new_y<0)
          new_y = new_y + in.rows;
          if (new_y>=in.rows)
          new_y = new_y - in.rows;

          for(x=0; x<in.cols; x++){

          // calculate new x-coordinate
          new_x = x + dx;
          if (new_x<0)
            new_x = new_x + in.cols;
          if (new_x>=in.cols)
            new_x = new_x - in.cols;
 
          tmp.at<Vec2f>(new_y, new_x) = in.at<Vec2f>(y, x);

        }
    }
    out = tmp;
}
