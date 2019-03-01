#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMask, int usFactor, double lambdaOverD);

