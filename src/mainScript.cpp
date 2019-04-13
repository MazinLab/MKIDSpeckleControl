#include "DMChannel.h"
#include "SpeckleToDM.h"
#include "SpeckleKalman.h"
#include "ImageGrabber.h"
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
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    
    //DMChannel chan("dm03disp03");
    //chan.getBufferPtr<float>();

    //SpeckleToDM speckInterface("dm04disp00");
    //speckInterface.addProbeSpeckle(cv::Point2d(20,20), 1, 0);
    //speckInterface.updateDM();
    //
    //boost::property_tree::ptree cfgParams;
    //read_info("speckNullConfig.info", cfgParams);

    //SpeckleKalman speck(cv::Point2d(10, 10), cfgParams);

    // CASE 1
    //DMChannel chan;
    //chan = DMChannel("dm04disp00");
    //chan.getXSize();//this segfaults for some reason!
    
    // CASE 2
    //DMChannel chan("dm04disp00");
    //chan.getXSize();

    // CASE 3
    //DMChannel *chan;
    //chan = new DMChannel("dm04disp00");
    //(*chan).getXSize();
    //delete chan;

}
