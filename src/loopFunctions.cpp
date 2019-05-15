#include "loopFunctions.h"

std::vector<int> loopfunctions::runLoop(int nIters, boost::property_tree::ptree &cfgParams, bool returnLC){
    std::vector<int> lightCurve;
    if(returnLC)
        lightCurve = std::vector<int>(nIters, 0);

    ImageGrabber imageGrabber(cfgParams.get_child("ImgParams"));
    SpeckleNuller nuller(cfgParams);
    nuller.updateBadPixMask(imageGrabber.getBadPixMaskCtrl());
    for(int i=0; i<nIters; i++){
        imageGrabber.startIntegrating(0, 2*cfgParams.get<double>("NullingParams.integrationTime"));
        cv::Mat &ctrlRegionImage = imageGrabber.getCtrlRegionImage();
        if(returnLC)
            lightCurve[i] = imageGrabber.getCtrlRegionCounts();
        nuller.update(ctrlRegionImage, cfgParams.get<double>("NullingParams.integrationTime"));
        nuller.updateDM();

    }

    return lightCurve;

}

std::vector<int> loopfunctions::runLoop(int nIters, const std::string cfgFilename, bool returnLC){
    boost::property_tree::ptree params;
    boost::property_tree::read_info(cfgFilename, params);

    return runLoop(nIters, params, returnLC);

}
