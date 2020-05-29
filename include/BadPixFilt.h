#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/log/trivial.hpp>

cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMask, int usFactor, double lambdaOverD);

