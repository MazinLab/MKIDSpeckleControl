#include "SpeckleKalman.h"

SpeckleKalman::SpeckleKalman(cv::Point2d pt, cv::Mat &image, boost::property_tree::ptree &ptree):
        SpeckleController(pt, image, ptree){
    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mProbeGridSpacing = mParams.get<double>("KalmanParams.probeGridSpacing");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;
    mKvecCorrSigma = 0.42*2*mProbeGridSpacing;
    mCurProbePos = cv::Point2i(mProbeGridWidth/2 + 1, mProbeGridWidth/2 + 1);
    mProbeAmp = calculateDMAmplitude(mKvecs, mInitialIntensity, mParams);

    mx = cv::Mat::zeros(2*mNProbePos, 1, CV_64F);
    mz = cv::Mat::zeros(2, 1, CV_64F);
    mK = cv::Mat::zeros(2*mNProbePos, 2, CV_64F);
    mH = cv::Mat::zeros(2, 2*mNProbePos, CV_64F);
    mA = cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mP = mParams.get<double>("KalmanParams.initStateVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQ = mParams.get<double>("KalmanParams.processNoiseVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mR = cv::Mat::zeros(2, 2, CV_64F);
    BOOST_LOG_TRIVIAL(info) << "SpeckleKalman: Initialized KF matrices";

    initializeProbeGridKvecs();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Probe Grid: " << mProbeGridKvecs;

    correlateProcessNoise();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Initial Process Noise: " << mP;


}

void SpeckleKalman::update(cv::Mat &image){
    std::tie(mPhaseIntensities[mCurPhaseInd], mPhaseSigmas[mCurPhaseInd]) = 
        measureSpeckleIntensityAndSigma(image);
     
    if(mCurPhaseInd == NPHASES-1){
        updateKalmanState();
        updateNullingSpeckle();
        mCurPhaseInd = 0;

    }
    else{
        mCurPhaseInd += 1;
        cv::Point2d curKvecs = mProbeGridKvecs.at<cv::Point2d>(mCurProbePos);
        mNextSpeck.kx = curKvecs.x;
        mNextSpeck.ky = curKvecs.y;
        mNextSpeck.amp = mProbeAmp;
        mNextSpeck.phase = mPhaseList[mCurPhaseInd];
        mNextSpeck.isNull = false;

    }

}

dmspeck SpeckleKalman::getNextSpeckle(){ 
    return mNextSpeck;

}

void SpeckleKalman::updateKalmanState(){
    int reInd, imInd;
    std::tie(reInd, imInd) = getKalmanIndices(mCurProbePos);
    mH.setTo(0);
    mH.at<double>(0, reInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    mH.at<double>(1, imInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    mz.at<double>(0) = mPhaseIntensities[0] - mPhaseIntensities[2];
    mz.at<double>(1) = mPhaseIntensities[1] - mPhaseIntensities[3];
    mR.at<double>(0,0) = std::pow(mPhaseSigmas[0], 2) + std::pow(mPhaseSigmas[2], 2);
    mR.at<double>(1,1) = std::pow(mPhaseSigmas[1], 2) + std::pow(mPhaseSigmas[3], 2);

    mx = mA*mx;
    mP = mA*mP*mA.t() + mQ;
    cv::Mat S = mR + mH*mP*mH.t();
    mK = mP*mH.t()*S.inv();
    mx = mx + mK*(mz - mH*mx);
    mP = (cv::Mat::eye(mP.rows, mP.cols, CV_64F) - mK*mH)*mP;

}

void SpeckleKalman::updateNullingSpeckle(){
    cv::Mat amplitude = cv::Mat::zeros(mNProbePos, mNProbePos, CV_64F);
    cv::Mat phase = cv::Mat::zeros(mNProbePos, mNProbePos, CV_64F);

    cv::Mat real(mx, cv::Range(0, mNProbePos));
    cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
    real = real.reshape(1, mProbeGridWidth);
    imag = imag.reshape(1, mProbeGridWidth);
    
    cv::cartToPolar(real, imag, amplitude, phase);

    cv::Mat weights = cv::Mat::ones(mNProbePos, mNProbePos, CV_64F);
    cv::GaussianBlur(weights, weights, cv::Size(0,0), 2*M_PI*0.42);
    weights = weights/cv::sum(weights)[0];
    cv::divide(amplitude, weights, weights);
    
    
}
    

void SpeckleKalman::correlateProcessNoise(){
    cv::Mat realBlock = cv::Mat(mP, cv::Range(0, mNProbePos), cv::Range(0, mNProbePos));
    cv::Mat imagBlock = cv::Mat(mP, cv::Range(mNProbePos, 2*mNProbePos), cv::Range(mNProbePos, 2*mNProbePos));
    cv::Point2d ki, kj;
    int row, col;
    double kDist;

    for(int i = 0; i < mNProbePos; i++){
        for(int j = 0; j < mNProbePos; j++){
            std::tie(row, col) = getProbeGridIndices(i);
            ki = mProbeGridKvecs.at<cv::Point2d>(row, col);
            std::tie(row, col) = getProbeGridIndices(j);
            kj = mProbeGridKvecs.at<cv::Point2d>(row, col);
            kDist = cv::norm(ki - kj);
            
            realBlock.at<double>(i, j) = std::sqrt(realBlock.at<double>(i, i)*realBlock.at<double>(j,j))*std::exp(-kDist*kDist/(2*mKvecCorrSigma*mKvecCorrSigma));
            imagBlock.at<double>(i, j) = std::sqrt(imagBlock.at<double>(i,i)*imagBlock.at<double>(j,j))*std::exp(-kDist*kDist/(2*mKvecCorrSigma*mKvecCorrSigma));

        }

    }

}

void SpeckleKalman::initializeProbeGridKvecs()
{
    mProbeGridKvecs = cv::Mat_<cv::Point2d>(mProbeGridWidth, mProbeGridWidth, cv::Point2d(0, 0));
    cv::Point2d kvec;
    
    for(int r = 0; r < mProbeGridWidth; r++){
        for(int c = 0; c < mProbeGridWidth; c++){
            kvec = cv::Point2d(mKvecs.x + mProbeGridSpacing*(c - mProbeGridWidth/2), 
                    mKvecs.y + mProbeGridSpacing*(r - mProbeGridWidth/2));
            mProbeGridKvecs.at<cv::Point2d>(r, c) = kvec;

        }

    }

}

std::tuple<int, int> SpeckleKalman::getKalmanIndices(int r, int c){
    int realInd = r*mProbeGridWidth + c;
    int imagInd = realInd + mNProbePos;
    return std::make_tuple(realInd, imagInd);

}

std::tuple<int, int> SpeckleKalman::getKalmanIndices(cv::Point2i &probePos){
    int realInd = probePos.y*mProbeGridWidth + probePos.x;
    int imagInd = realInd + mNProbePos;
    return std::make_tuple(realInd, imagInd);

}

std::tuple<int, int> SpeckleKalman::getProbeGridIndices(int kalmanInd)
{
    int row = kalmanInd/mProbeGridWidth;
    int col = kalmanInd%mProbeGridWidth;
    return std::make_tuple(row, col);

}

