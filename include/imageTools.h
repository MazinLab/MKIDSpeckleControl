#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/log/trivial.hpp>

cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMask, cv::Mat &kernel, int usFactor);

cv::Mat createGaussianFilter(int usFactor, double lambdaOverD);

