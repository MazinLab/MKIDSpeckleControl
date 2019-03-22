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

#ifndef DMTOOLS_H
#define DMTOOLS_H
/**
 * Calculates speckle k-vectors (spatial frequencies) from coordinates. PSF location is provided by the 
 * config params.
 * @param coords Speckle coordinates within the control region
 * @param ctrlRegionCoords Coordinates of the control region (top left corner) relative to PSF
 * @param dmAngle Angle between DM k-vector basis and image x-y axes
 * @param lambdaOverD N pixels per lambda/D
 * @param yFlip Set to true if there's a reflection between DM and image
 */
cv::Point2d calculateKvecs(const cv::Point2d &coords, const cv::Point2d &ctrlRegionCoords, double dmAngle, double lambdaOverD, bool yFlip);

/**
 * Overloaded function.
 * @param coords Speckle coordinates within the control region
 * @param cfgParams Configuration parameters. Relevant ones are lambda/D and PSF location on the array
 */
cv::Point2d calculateKvecs(const cv::Point2d &coords, boost::property_tree::ptree &cfgParams);


/**
 * Calculates DM amplitude according to: 
 * sqrt(1000*intensity*(a|k|^2 + b|k| + c)/integrationTime)
 * @param kvecs Speckle k-vectors
 * @param intensity Speckle intensity (# of photons)
 * @param integrationTime Integration time in ms
 * @param a, b, c Calibration coefficients
 */
double calculateDMAmplitude(const cv::Point2d &kvecs, double intensity, double integrationTime, double a, double b, double c);

/**
 * Overloaded function. Calculates DM amplitude from calibration.
 * @param kvecs Speckle k-vectors
 * @param intensity Speckle intensity
 */
double calculateDMAmplitude(const cv::Point2d &kvecs, double intensity, boost::property_tree::ptree &cfgParams);

#endif
