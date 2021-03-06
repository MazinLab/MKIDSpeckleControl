#include "dmTools.h"

cv::Point2d calculateKvecs(const cv::Point2d &coords, boost::property_tree::ptree &cfgParams)
{
    cv::Point2d intCoords, kvecs;
    double dmAngle = cfgParams.get<double>("DMParams.angle");
    intCoords.x = (double)(coords.x + cfgParams.get<int>("ImgParams.xCtrlStart"));
    intCoords.y = (double)(coords.y + cfgParams.get<int>("ImgParams.yCtrlStart"));
    kvecs.x = 2.0*M_PI*(std::cos(-dmAngle)*intCoords.x - std::sin(-dmAngle)*intCoords.y)/cfgParams.get<double>("ImgParams.lambdaOverD");
    kvecs.y = cfgParams.get<double>("DMParams.yFlip")*2.0*M_PI*(std::sin(-dmAngle)*intCoords.x + std::cos(-dmAngle)*intCoords.y)/cfgParams.get<double>("ImgParams.lambdaOverD");
    return kvecs;

}

cv::Point2d calculateKvecs(const cv::Point2d &coords, const cv::Point2d &ctrlRegionCoords, double dmAngle, double lambdaOverD, bool yFlip){
    cv::Point2d intCoords, kvecs;
    double yFactor = 1;
    if(yFlip)
        yFactor = -1;
    intCoords = coords + ctrlRegionCoords;
    kvecs.x = 2.0*M_PI*(std::cos(-dmAngle)*intCoords.x - std::sin(-dmAngle)*intCoords.y)/lambdaOverD;
    kvecs.y = yFactor*2.0*M_PI*(std::sin(-dmAngle)*intCoords.x + std::cos(-dmAngle)*intCoords.y)/lambdaOverD;
    return kvecs;

}

double calculateDMAmplitude(const cv::Point2d &kvecs, double intensity, boost::property_tree::ptree &cfgParams)
{
    double k = cv::norm(kvecs);
    return std::sqrt(1000*intensity/cfgParams.get<double>("NullingParams.integrationTime")*(cfgParams.get<double>("DMParams.a")*k*k + cfgParams.get<double>("DMParams.b")*k + cfgParams.get<double>("DMParams.c")));

}

double calculateDMAmplitude(const cv::Point2d &kvecs, double intensity, double integrationTime, double a, double b, double c){
    double k = cv::norm(kvecs);
    return std::sqrt(1000*intensity/integrationTime*(a*k*k + b*k + c));

}

double getDMCalFactor(const cv::Point2d &kvecs, double integrationTime, double a, double b, double c){
    double k = cv::norm(kvecs);
    return std::sqrt(1000/integrationTime*(a*k*k + b*k + c));

}

double getDMCalFactorCPS(const cv::Point2d &kvecs, double a, double b, double c){
    double k = cv::norm(kvecs);
    return std::sqrt((a*k*k + b*k + c));

}
