#include "DMChannel.h"
#include "SpeckleToDM.h"
#include <opencv2/opencv.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

int main()
{ 
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    //DMChannel chan("dm03disp03");
    //chan.getBufferPtr<uint8_t>();
    SpeckleToDM speckInterface("dm03disp09");
    speckInterface.addProbeSpeckle(cv::Point2d(20,20), 1, 0);
    speckInterface.updateDM();
    
    //DMChannel chan;
    //chan = DMChannel("dm00disp00") //this segfaults for some reason!
    DMChannel chan("dm00disp00");
    chan.getXSize();

}
