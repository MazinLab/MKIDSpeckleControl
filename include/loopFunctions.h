#include "SpeckleNuller.h"
#include "ImageGrabber.h"
#include <opencv2/opencv.hpp>

//logging
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

//cfg parser
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

namespace loopfunctions{

std::vector<int> runLoop(int nIters, boost::property_tree::ptree &cfgParams, bool returnLC=false);

std::vector<int> runLoop(int nIters, const std::string cfgFilename, bool returnLC=false);

}
