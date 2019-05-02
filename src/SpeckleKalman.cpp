#include "SpeckleKalman.h"

SpeckleKalman::SpeckleKalman(cv::Point2d pt, boost::property_tree::ptree &ptree):
        SpeckleController(pt, ptree){
    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mProbeGridSpacing = mParams.get<double>("KalmanParams.probeGridSpacing");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;
    mKvecCorrSigma = 0.42*2*mProbeGridSpacing;
    mCurProbePos = cv::Point2i(mProbeGridWidth/2, mProbeGridWidth/2);

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

void SpeckleKalman::update(const cv::Mat &image){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": mCurPhaseInd: " << mCurPhaseInd;
    if(mCurPhaseInd == -1){
        BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": mCurProbePos: " << mCurProbePos;
        double intensity, sigma;
        std::tie(intensity, sigma) = measureSpeckleIntensityAndSigma(image);
        nonProbeMeasurmentUpdate(intensity, sigma);

    }

    else
        std::tie(mPhaseIntensities[mCurPhaseInd], mPhaseSigmas[mCurPhaseInd]) = 
                measureSpeckleIntensityAndSigma(image);
         
   if(mCurPhaseInd == NPHASES-1){
       updateKalmanState();
       updateNullingSpeckle();
       mCurPhaseInd = -1;

   }

   else{
       cv::Point2d curKvecs = mProbeGridKvecs.at<cv::Point2d>(mCurProbePos);
       mNextSpeck.kx = curKvecs.x;
       mNextSpeck.ky = curKvecs.y;
       mNextSpeck.amp = mProbeAmp;
       mNextSpeck.phase = mPhaseList[mCurPhaseInd];
       mNextSpeck.isNull = false;
       mCurPhaseInd += 1;

   }

    

}

dmspeck SpeckleKalman::getNextSpeckle() const{ 
    return mNextSpeck;

}

void SpeckleKalman::nonProbeMeasurmentUpdate(double intensity, double sigma){
    mInitialIntensity = intensity;
    mInitialSigma = sigma;
    mProbeAmp = calculateDMAmplitude(mKvecs, mInitialIntensity, mParams);

}
    

void SpeckleKalman::updateKalmanState(){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": updating state estimate";
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
    cv::Mat amplitude = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::Mat phase = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": calculating nulling speckle";

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Amplitude: " << amplitude;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Phase: " << phase;

    cv::Mat real(mx, cv::Range(0, mNProbePos));
    cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
    real = real.reshape(1, mProbeGridWidth);
    imag = imag.reshape(1, mProbeGridWidth);
    
    cv::Mat variance = mP.diag();
    variance = cv::Mat(variance, cv::Range(0, mNProbePos)) + cv::Mat(variance, cv::Range(mNProbePos, 2*mNProbePos));
    variance = variance.reshape(1, mProbeGridWidth);
    assert(variance.cols == mProbeGridWidth);
    
    cv::cartToPolar(real, imag, amplitude, phase);

    // weights_ij is proportional to amplitude_ij/(overlap of probe grid w/ gaussian)
    cv::Mat weights = cv::Mat::ones(mNProbePos, mNProbePos, CV_64F);
    cv::GaussianBlur(weights, weights, cv::Size(0,0), 2*M_PI*0.42);
    cv::divide(amplitude, weights, weights);
    weights = weights/cv::sum(weights)[0];    

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: weights " << weights;

    cv::Point2d nullingK;
    double nullingAmp;
    double nullingPhase;
    double nullingVar;
    mProbeGridKvecs.forEach([this, &weights, &amplitude, &phase, &nullingK, &nullingAmp, &nullingPhase]
                (cv::Point2d &value, const int *position) -> void{
            nullingK += weights.at<double>(position[0], position[1])*value;
            nullingAmp += weights.at<double>(position[0], position[1])*amplitude.at<double>(position[0], position[1]);
            nullingPhase += weights.at<double>(position[0], position[1])*phase.at<double>(position[0], position[1]); 

            });

    nullingVar = cv::sum(variance.mul(weights))[0]; //this could also be done inside lambda function

    double snr = nullingAmp/std::sqrt(nullingVar);

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling k: " << nullingK;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling amplitude: " << nullingAmp;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling phase: " << nullingPhase;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling variance: " << nullingVar;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling snr: " << snr;

    mNextSpeck.kx = nullingK.x;
    mNextSpeck.ky = nullingK.y;
    mNextSpeck.phase = nullingPhase;
    mNextSpeck.isNull = true;

    if(snr >= mParams.get<double>("KalmanParams.snrThresh"))
        mNextSpeck.amp = nullingAmp;

    else
        mNextSpeck.amp = 0;
    
    
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

