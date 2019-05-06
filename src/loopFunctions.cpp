#include "loopFunctions.h"

void loopfunctions::runLoop(int nIters, boost::property_tree::ptree &cfgParams){
    ImageGrabber imageGrabber(cfgParams.get_child("ImgParams"));
    SpeckleNuller nuller(cfgParams);
    nuller.updateBadPixMask(imageGrabber.getBadPixMaskCtrl());
    for(int i=0; i<nIters; i++){
        imageGrabber.startIntegrating(0, 2*cfgParams.get<double>("NullingParams.integrationTime"));
        cv::Mat &ctrlRegionImage = imageGrabber.getCtrlRegionImage();
        nuller.update(ctrlRegionImage, cfgParams.get<double>("NullingParams.integrationTime"));
        nuller.updateDM();

    }

}

void loopfunctions::runLoop(int nIters, const std::string cfgFilename){
    boost::property_tree::ptree params;
    boost::property_tree::read_info(cfgFilename, params);

    runLoop(nIters, params);

}
