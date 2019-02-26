#include "dmTools.h"

cv::Point2d calculateKVecs(const cv::Point2d &coords, boost::property_tree::ptree &cfgParams)
{
    cv::Point2d intCoords, kvecs;
    double dmAngle = cfgParams.get<double>("DMCal.angle");
    intCoords.x = (double)(coords.x + cfgParams.get<int>("ImgParams.xCtrlStart"));
    intCoords.y = (double)(coords.y + cfgParams.get<int>("ImgParams.yCtrlStart"));
    kvecs.x = 2.0*M_PI*(std::cos(-dmAngle)*intCoords.x - std::sin(-dmAngle)*intCoords.y)/cfgParams.get<double>("ImgParams.lambdaOverD");
    kvecs.y = cfgParams.get<double>("DMCal.yFlip")*2.0*M_PI*(std::sin(-dmAngle)*intCoords.x + std::cos(-dmAngle)*intCoords.y)/cfgParams.get<double>("ImgParams.lambdaOverD");
    return kvecs;

}

double calculateDMAmplitude(const cv::Point2d &kvecs, unsigned short intensity, boost::property_tree::ptree &cfgParams)
{
    double k = norm(kvecs);
    return std::sqrt(1000*intensity/cfgParams.get<double>("ImgParams.integrationTime")*(cfgParams.get<double>("DMCal.a")*k*k + cfgParams.get<double>("DMCal.b")*k + cfgParams.get<double>("DMCal.c")));

}

