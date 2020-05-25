#include "SpeckleNuller.h"

bool cmpImgPt(ImgPt lhs, ImgPt rhs)
{
    return lhs.intensity > rhs.intensity;

}

SpeckleNuller::SpeckleNuller(boost::property_tree::ptree &ptree) : 
        mDM(ptree.get<std::string>("DMParams.channel").c_str()), mIters(0){
    mParams = ptree;
    int ctrlRegionXSize = mParams.get<int>("ImgParams.xCtrlEnd") - mParams.get<int>("ImgParams.xCtrlStart");
    int ctrlRegionYSize = mParams.get<int>("ImgParams.yCtrlEnd") - mParams.get<int>("ImgParams.yCtrlStart");
    mImage.create(ctrlRegionYSize, ctrlRegionXSize, CV_64FC1);
    mBadPixMask.create(ctrlRegionXSize, ctrlRegionYSize, CV_64F);
    mBadPixMask.setTo(0);
    mSpecklesList.reserve(mParams.get<int>("NullingParams.maxSpeckles"));
    if(mParams.get<std::string>("NullingParams.controller") == "basic"){
        mParams.put("NullingParams.maxNullingIters", 1);
        BOOST_LOG_TRIVIAL(info) << "Using basic controller, setting max nulling iters to 1";

    }

}

void SpeckleNuller::update(const cv::Mat &newImage, double integrationTime){
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: updating...";
    mIntegrationTime = integrationTime;
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: convert image to float...";
    newImage.convertTo(mImage, CV_64F);
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: done convert image to float...";
    updateSpeckles();
    if(mIters%(NPHASES + 1) == 0)
        findNewSpeckles();
    mIters++;
    

}

void SpeckleNuller::updateBadPixMask(const cv::Mat &newMask){
    mBadPixMask = newMask;
    boost::ptr_vector<SpeckleController>::iterator it;

    for(it = mSpecklesList.begin(); it < mSpecklesList.end(); it++)
        it->updateBadPixMask(newMask);

}



std::vector<ImgPt> SpeckleNuller::detectSpeckles(){ 
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: detecting new speckles...";
    double usFactor = mParams.get<double>("NullingParams.usFactor");

    //first do gaussian us filt on image
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: gaussian filtering...";
    cv::Mat filtImg = gaussianBadPixUSFilt(mImage, mBadPixMask, (int)usFactor, mParams.get<double>("ImgParams.lambdaOverD"));
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: done filtering...";

    //scale image parameters by usFactor, since image is upsampled
    int speckleWindow = mParams.get<int>("NullingParams.speckleWindow")*mParams.get<int>("NullingParams.usFactor");
    int apertureRadius = mParams.get<double>("NullingParams.apertureRadius");
    //BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: params: " << speckleWindow << " " << apertureRadius;

    //Find local maxima within mParams.get<int>("NullingParams.speckleWindow") size window
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: finding local maxima...";
    cv::Mat kernel = cv::Mat::ones(speckleWindow, speckleWindow, CV_8UC1);
    cv::Mat maxFiltIm, isMaximum;
    std::vector<cv::Point2i> maxima;
    std::vector<cv::Point2i> speckleLocs;
    std::vector<ImgPt> maxImgPts;

    if(mParams.get<bool>("NullingParams.useBoxBlur"))
        cv::blur(filtImg.clone(), filtImg, cv::Size2i(speckleWindow, speckleWindow));
    
    if(mParams.get<bool>("NullingParams.useGaussianBlur"))
        cv::blur(filtImg.clone(), filtImg, cv::Size2i(speckleWindow, speckleWindow));

    //BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: dilating...";
    cv::dilate(filtImg, maxFiltIm, kernel);
    //BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: marking maxima...";
    cv::compare(filtImg, maxFiltIm, isMaximum, cv::CMP_EQ);
    //BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: finding nonzero...";
    isMaximum = isMaximum & (filtImg != 0);
    cv::findNonZero(isMaximum, maxima); //maxima are coordinates in upsampled filtImg
    BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: found " << maxima.size() << " local maxima";
    
    //Put Points in ImgPt Struct List
    std::vector<cv::Point2i>::iterator it;
    ImgPt tempPt;
    for(it = maxima.begin(); it != maxima.end(); it++)
    {
        tempPt.coordinates = cv::Point2d((double)(*it).x/usFactor, (double)(*it).y/usFactor); //coordinates in real image
        tempPt.intensity = filtImg.at<double>(*it);
        BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: Detected speckle at " << tempPt.coordinates << " intensity: " << tempPt.intensity;
        if(tempPt.intensity != 0)
            if((tempPt.coordinates.x < (mImage.cols-apertureRadius)) && (tempPt.coordinates.x > (apertureRadius))
                && (tempPt.coordinates.y < (mImage.rows-apertureRadius)) && (tempPt.coordinates.y > (apertureRadius)))
            maxImgPts.push_back(tempPt);

    }
    
    //Sort list of ImgPts
    std::sort(maxImgPts.begin(), maxImgPts.end(), cmpImgPt);

    BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: Detected " << maxImgPts.size() << " bright spots.";
    
    
    //imgGrabber.displayImage(true);
    
    return maxImgPts;

}

void SpeckleNuller::exclusionZoneCut(std::vector<ImgPt> &maxImgPts, bool checkCurrentSpeckles)
{
    std::vector<ImgPt>::iterator curElem, kt;
    ImgPt curPt;
    double ptDist;
    bool curElemRemoved;
    int exclusionZone = mParams.get<int>("NullingParams.exclusionZone");

    for(curElem = maxImgPts.begin(); 
        (curElem < (maxImgPts.begin()+mParams.get<int>("NullingParams.maxSpeckles"))) && (curElem < maxImgPts.end()); curElem++)
    {
        curPt = *curElem;
        curElemRemoved = false;
        //Check to see if curPt is too close to any speckle, but only if we're keeping all currently active speckles
        if(checkCurrentSpeckles)
        {
            boost::ptr_vector<SpeckleController>::iterator speckIter;
            for(speckIter = mSpecklesList.begin(); speckIter < mSpecklesList.end(); speckIter++)
            {
                ptDist = cv::norm(curPt.coordinates - (*speckIter).getCoordinates());
                if(ptDist <= exclusionZone)
                {
                    maxImgPts.erase(curElem);
                    curElem--;
                    curElemRemoved = true;
                    break;

                }

            }

        }

        if(curElemRemoved)
            continue;

        //If not too close to speckle, remove other maxImgPts within exclusion zone
        for(kt = curElem+1; kt < maxImgPts.end(); kt++)
        {
            ptDist = cv::norm(curPt.coordinates - (*kt).coordinates);
            // BOOST_LOG_TRIVIAL(debug) << "curElem " << (*curElem).coordinates;
            // BOOST_LOG_TRIVIAL(debug) << "maxImgPts0" << maxImgPts[0].coordinates;
            // BOOST_LOG_TRIVIAL(debug) << "kt " << (*kt).coordinates;
            // BOOST_LOG_TRIVIAL(debug) << "maxImgPtsend " << (*(maxImgPts.end()-1)).coordinates;
            if(ptDist <= exclusionZone)
            {
                //BOOST_LOG_TRIVIAL(debug) << "pdist" << ptDist;
                maxImgPts.erase(kt);
                kt--;

            }

        }

    }

    //if(mSpecklesList.size() >= mParams.get<int>("NullingParams.maxSpeckles"))
    //    maxImgPts.clear();
    //
    //else if(maxImgPts.size() > (mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size()))
    //    maxImgPts.erase(maxImgPts.begin()+mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size(), maxImgPts.end());
    


}

void SpeckleNuller::updateAndCutActiveSpeckles(std::vector<ImgPt> &maxImgPts){
    std::vector<ImgPt>::iterator ptIter;
    boost::ptr_vector<SpeckleController>::iterator speckIter;
    bool speckFound; // check if nulled speckle has been detected
    double ptDist;
    
    for(speckIter = mSpecklesList.begin(); speckIter < mSpecklesList.end(); speckIter++){
        speckFound = false;
        for(ptIter = maxImgPts.begin(); ptIter < maxImgPts.end(); ptIter++){
            ptDist = cv::norm((*ptIter).coordinates - (*speckIter).getCoordinates());
            if(ptDist < mParams.get<double>("NullingParams.distThresh")){
                speckFound = true;
                //if(mParams.get<bool>("TrackingParams.updateCoords"))
                //    (*speckIter).setCoordinates((*ptIter).coordinates);
                //(*speckIter).measureIntensityCorrection(mBadPixMask);
                //(*speckIter).measureSpeckleIntensityAndSigma(mImage);
                BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: Re-detected active speckle at " << (*ptIter).coordinates;
                maxImgPts.erase(ptIter);
                break;

            }


        }

        if(!speckFound){ // delete speckle from mSpecklesList if it wasn't detected this iteration
            BOOST_LOG_TRIVIAL(info) << "SpeckleNuller: Failed to detect active speckle at " 
                << speckIter->getCoordinates() << "; deleting.";
            mSpecklesList.erase(speckIter);
            speckIter--;

        }

    }

}

void SpeckleNuller::updateAndCutNulledSpeckles(std::vector<ImgPt> &maxImgPts){
    std::vector<ImgPt>::iterator ptIter;
    boost::ptr_vector<SpeckleController>::iterator speckIter;
    bool speckFound; // check if nulled speckle has been detected
    int exclusionZone = mParams.get<int>("NullingParams.exclusionZone");
    double ptDist;
    
    for(speckIter = mNullSpecklesList.begin(); speckIter < mNullSpecklesList.end(); speckIter++){
        speckFound = false;
        for(ptIter = maxImgPts.begin(); ptIter < maxImgPts.end(); ptIter++){
            ptDist = cv::norm((*ptIter).coordinates - (*speckIter).getCoordinates());
            if(ptDist < mParams.get<double>("NullingParams.distThresh")){
                speckFound = true;
                //if(mParams.get<bool>("TrackingParams.updateCoords"))
                //    (*speckIter).setCoordinates((*ptIter).coordinates);
                BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: Re-detected nulled speckle at " << (*ptIter).coordinates;
                mSpecklesList.push_back(&(*speckIter)); //OMG
                maxImgPts.erase(ptIter);
                break;

            }

        }

    }

    mNullSpecklesList.clear();

}
            
void SpeckleNuller::createSpeckleObjects(std::vector<ImgPt> &imgPts, bool update){ 
    BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: creating speckle objects...";

    std::vector<ImgPt>::iterator it;
    cv::Point2d coordinates;
    SpeckleController *speck;
    for(it = imgPts.begin(); it < imgPts.end(); it++){
        coordinates = (*it).coordinates;
        if(mParams.get<std::string>("NullingParams.controller") == "kalman")
            speck = new SpeckleKalman(coordinates, mParams); 
        else if(mParams.get<std::string>("NullingParams.controller") == "kalmanPoisson")
            speck = new SpeckleKalmanPoisson(coordinates, mParams);
        else if(mParams.get<std::string>("NullingParams.controller") == "basic")
            speck = new SpeckleBasic(coordinates, mParams);
        else
            throw "Controller type not implemented";
        speck->updateBadPixMask(mBadPixMask);
        if(update){
            speck->update(mImage, mIntegrationTime);
            mNextDMSpecks.push_back(speck->getNextSpeckle());

        }

        mSpecklesList.push_back(speck);

    }


}

void SpeckleNuller::updateSpeckles(){
    boost::ptr_vector<SpeckleController>::iterator it;
    dmspeck speck;

    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: updating speckle objects...";
    
    for(it = mSpecklesList.begin(); it < mSpecklesList.end(); it++){
        it->update(mImage, mIntegrationTime);
        speck = it->getNextSpeckle();

        if(speck.amp != 0)
            mNextDMSpecks.push_back(speck);

        if(it->getNNullingIters() >= mParams.get<int>("NullingParams.maxNullingIters")){
            BOOST_LOG_TRIVIAL(info) << "Deleting nulled speckle at " << it->getCoordinates() << " after " 
                << it->getNProbeIters() << " probe iters.";
            mSpecklesList.erase(it);
            it--;

        }

        else if(it->getNProbeIters() >= mParams.get<int>("NullingParams.maxProbeIters")){
            BOOST_LOG_TRIVIAL(info) << "Deleting speckle at " << it->getCoordinates() << " after " 
                << it->getNProbeIters() << " probe iters.";
            mSpecklesList.erase(it);
            it--;

        }

            

    }

    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: done updating speckle objects";
        
}

void SpeckleNuller::findNewSpeckles(){
    std::vector<ImgPt> imgPts;
    imgPts = detectSpeckles();
    
    if(mParams.get<bool>("NullingParams.enforceRedetection")){
        exclusionZoneCut(imgPts, false);
        if(imgPts.size() > mParams.get<int>("NullingParams.maxSpeckles"))
            imgPts.erase(imgPts.begin() + mParams.get<int>("NullingParams.maxSpeckles"), imgPts.end());

        updateAndCutActiveSpeckles(imgPts);

    }

    else{
        BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: start exclusionZoneCut";
        exclusionZoneCut(imgPts, true);
        if(imgPts.size() > (mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size()))
            imgPts.erase(imgPts.begin() + mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size(), imgPts.end());
        BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: done exclusionZoneCut";

    }

    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: creating speckle objects";
    createSpeckleObjects(imgPts);
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: done creating speckle objects";

}

void SpeckleNuller::updateDM(){
    std::vector<dmspeck>::iterator it;
    mDM.clearProbeSpeckles();

    for(it = mNextDMSpecks.begin(); it < mNextDMSpecks.end(); it++){
        if(it->isNull)
            mDM.addNullingSpeckle(it->kx, it->ky, it->amp, it->phase);
        else
            mDM.addProbeSpeckle(it->kx, it->ky, it->amp, it->phase);

    }

    mDM.updateDM();
    mNextDMSpecks.clear();

}
 
        
void SpeckleNuller::clearSpeckleObjects() {mSpecklesList.clear();}
