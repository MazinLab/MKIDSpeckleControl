#include "loopFunctions.h"

void loopfunctions::runLoop(int nIters, boost::property_tree::ptree &cfgParams){
    ImageGrabber imageGrabber(cfgParams.get_child("ImageParams"));
    SpeckleNuller nuller(cfgParams);
    for(int i=0; i<nIters; i++){
        imageGrabber.startIntegrating(0, cfgParams.get<double>("NullingParams.integrationTime"));
        cv::Mat &ctrlRegionImage = imageGrabber.getCtrlRegionImage();
        nuller.update(ctrlRegionImage);

    }

}
