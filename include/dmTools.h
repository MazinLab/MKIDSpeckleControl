/**
* dmTools: set of functions for generating/converting flatmaps, centoffs, etc.
**/

#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <iostream>
#include "params.h"
#include <boost/property_tree/ptree.hpp>

typedef double Pixel;

/**
* Calculates speckle k-vectors (spatial frequencies) from coordinates. PSF location is provided by the 
* config params.
* @param coords Speckle coordinates on array
* @param cfgParams Configuration parameters. Relevant ones are lambda/D and PSF location on the array
**/
cv::Point2d calculateKVecs(const cv::Point2d &coords, boost::property_tree::ptree &cfgParams);


/**
* Calculates DM amplitude from calibration.
* @param kvecs Speckle k-vectors
* @param intensity Speckle intensity
**/
double calculateDMAmplitude(const cv::Point2i &kvecs, unsigned short intensity, boost::property_tree::ptree &cfgParams);

