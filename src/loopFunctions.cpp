#include "loopFunctions.h"

void runLoop(int nIters, boost::property_tree::ptree &cfgParams){
    ImageGrabber imageGrabber(cfgParams.get_child("ImageParams"));
    SpeckleNuller nuller(cfgParams);

}
