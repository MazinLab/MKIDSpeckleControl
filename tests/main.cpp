#include "DMChannel.h"
#include "SpeckleToDM.h"
#include "SpeckleKalman.h"
#include "ImageGrabber.h"
#include "loopFunctions.h"
#include "dmTools.h"
#include <opencv2/opencv.hpp>

//logging
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

//cfg parser
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

int main()
{ 
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
    boost::property_tree::ptree cfgParams;
    read_info("python/speckNullConfig.info", cfgParams);

    loopfunctions::runLoop(10000, cfgParams);
    //DMChannel chan("dm04disp00");
    //chan.save("dm04disp00");
    


}
