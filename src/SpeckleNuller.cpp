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

}

void SpeckleNuller::update(const cv::Mat &newImage, double integrationTime){
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: updating...";
    mIntegrationTime = integrationTime;
    newImage.convertTo(mImage, CV_64F);
    if(mIters%(NPHASES + 1) == 0)
        findNewSpeckles();
    updateSpeckles();
    mIters++;
    

}

void SpeckleNuller::updateBadPixMask(const cv::Mat &newMask){
    mBadPixMask = newMask;
    std::vector<SpeckleCtrlClass>::iterator it;

    for(it = mSpecklesList.begin(); it < mSpecklesList.end(); it++)
        it->updateBadPixMask(newMask);

}



std::vector<ImgPt> SpeckleNuller::detectSpeckles(){ 
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: detecting new speckles...";
    double usFactor = mParams.get<double>("NullingParams.usFactor");

    //first do gaussian us filt on image
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: gaussian filtering...";
    cv::Mat filtImg = gaussianBadPixUSFilt(mImage, mBadPixMask, (int)usFactor, mParams.get<double>("ImgParams.lambdaOverD"));

    //scale image parameters by usFactor, since image is upsampled
    int speckleWindow = mParams.get<int>("NullingParams.speckleWindow")*mParams.get<int>("NullingParams.usFactor");
    int apertureRadius = mParams.get<double>("NullingParams.apertureRadius");
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: params: " << speckleWindow << " " << apertureRadius;

    //Find local maxima within mParams.get<int>("NullingParams.speckleWindow") size window
    cv::Mat kernel = cv::Mat::ones(speckleWindow, speckleWindow, CV_8UC1);
    cv::Mat maxFiltIm, isMaximum;
    std::vector<cv::Point2i> maxima;
    std::vector<cv::Point2i> speckleLocs;
    std::vector<ImgPt> maxImgPts;

    if(mParams.get<bool>("NullingParams.useBoxBlur"))
        cv::blur(filtImg.clone(), filtImg, cv::Size2i(speckleWindow, speckleWindow));
    
    if(mParams.get<bool>("NullingParams.useGaussianBlur"))
        cv::blur(filtImg.clone(), filtImg, cv::Size2i(speckleWindow, speckleWindow));

    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: dilating...";
    cv::dilate(filtImg, maxFiltIm, kernel);
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: marking maxima...";
    cv::compare(filtImg, maxFiltIm, isMaximum, cv::CMP_EQ);
    BOOST_LOG_TRIVIAL(trace) << "SpeckleNuller: finding nonzero...";
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

void SpeckleNuller::exclusionZoneCut(std::vector<ImgPt> &maxImgPts)
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
        if(!mParams.get<bool>("NullingParams.enforceRedetection"))
        {
            std::vector<SpeckleCtrlClass>::iterator speckIter;
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

    if(mSpecklesList.size() >= mParams.get<int>("NullingParams.maxSpeckles"))
        maxImgPts.clear();
    
    else if(maxImgPts.size() > (mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size()))
        maxImgPts.erase(maxImgPts.begin()+mParams.get<int>("NullingParams.maxSpeckles") - mSpecklesList.size(), maxImgPts.end());


}

void SpeckleNuller::updateAndCutActiveSpeckles(std::vector<ImgPt> &maxImgPts){
    std::vector<ImgPt>::iterator ptIter;
    std::vector<SpeckleCtrlClass>::iterator speckIter;
    bool speckFound; // check if nulled speckle has been detected
    int exclusionZone = mParams.get<int>("NullingParams.exclusionZone");
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
            mSpecklesList.erase(speckIter);
            speckIter--;

        }

    }

}

void SpeckleNuller::updateAndCutNulledSpeckles(std::vector<ImgPt> &maxImgPts){
    std::vector<ImgPt>::iterator ptIter;
    std::vector<SpeckleCtrlClass>::iterator speckIter;
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
                mSpecklesList.push_back(*speckIter);
                maxImgPts.erase(ptIter);
                break;

            }

        }

    }

    mNullSpecklesList.clear();

}
            
void SpeckleNuller::createSpeckleObjects(std::vector<ImgPt> &imgPts){ 
    BOOST_LOG_TRIVIAL(debug) << "SpeckleNuller: creating speckle objects...";

    std::vector<ImgPt>::iterator it;
    cv::Point2d coordinates;
    for(it = imgPts.begin(); it < imgPts.end(); it++){
        coordinates = (*it).coordinates;
        SpeckleCtrlClass speck(coordinates, mParams);
        mSpecklesList.push_back(speck);

    }


}

void SpeckleNuller::updateSpeckles(){
    std::vector<SpeckleCtrlClass>::iterator it;
    dmspeck speck;
    
    for(it = mSpecklesList.begin(); it < mSpecklesList.end(); it++){
        it->update(mImage, mIntegrationTime);
        speck = it->getNextSpeckle();
        if(speck.amp == DELETE_SPECKLE){
            mSpecklesList.erase(it);
            it--;

        }

        else if(speck.amp == 0)
            continue;

        else
            mNextDMSpecks.push_back(speck);

    }
        
}

void SpeckleNuller::findNewSpeckles(){
    std::vector<ImgPt> imgPts;
    imgPts = detectSpeckles();
    
    if(mParams.get<bool>("NullingParams.enforceRedetection"))
        updateAndCutActiveSpeckles(imgPts);

    exclusionZoneCut(imgPts);
    createSpeckleObjects(imgPts);

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
