#include "loopFunctions.h"

std::vector<int> loopfunctions::runLoop(int nIters, boost::property_tree::ptree &cfgParams, bool returnLC, 
        bool useAbsTiming){
    std::vector<int> lightCurve;
    if(returnLC)
        lightCurve = std::vector<int>(nIters, 0);

    ImageGrabber imageGrabber(cfgParams.get_child("ImgParams"));
    SpeckleNuller nuller(cfgParams);
    std::chrono::microseconds rawTime;
    uint64_t timestamp;

    nuller.updateBadPixMask(imageGrabber.getBadPixMaskCtrl());
    for(int i=0; i<nIters; i++){
        if(useAbsTiming){
            rawTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());                                                                      
            timestamp = rawTime.count()/500 - (uint64_t)TSOFFS*2000;

        }
        else
            timestamp = 0;
        imageGrabber.startIntegrating(timestamp, cfgParams.get<double>("NullingParams.integrationTime"));
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
