#include "SpeckleKalman.h"

SpeckleKalman::SpeckleKalman(cv::Point2d pt, boost::property_tree::ptree &ptree):
        SpeckleController(pt, ptree){
    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mProbeGridSpacing = mParams.get<double>("KalmanParams.probeGridSpacing");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;
    mKvecCorrSigma = 0.42*2*mProbeGridSpacing;
    mCurProbePos = cv::Point2i(mProbeGridWidth/2, mProbeGridWidth/2);
    mDMCalFactor = getDMCalFactorCPS(mKvecs, mParams.get<double>("DMParams.a"), mParams.get<double>("DMParams.b"), 
            mParams.get<double>("DMParams.c"));
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: DMCalFactor: " << mDMCalFactor;

    mx = cv::Mat::zeros(2*mNProbePos, 1, CV_64F);
    mz = cv::Mat::zeros(2, 1, CV_64F);
    mK = cv::Mat::zeros(2*mNProbePos, 2, CV_64F);
    mH = cv::Mat::zeros(2, 2*mNProbePos, CV_64F);
    mA = cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mP = mParams.get<double>("KalmanParams.initStateVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQ = mParams.get<double>("KalmanParams.processNoiseVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQc = cv::Mat::zeros(2*mNProbePos, 2*mNProbePos, CV_64F);
    mR = cv::Mat::zeros(2, 2, CV_64F);
    mMinProbeAmp = 2;

    initializeProbeGridKvecs();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Probe Grid: " << mProbeGridKvecs;

    correlateProcessNoise(mP);
    //BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Initial State Variance: " << mP;

    mMinProbeIters = mParams.get<int>("KalmanParams.minProbeIters");


}

void SpeckleKalman::update(const cv::Mat &image, double integrationTime){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": mCurPhaseInd: " << mCurPhaseInd;
    if(mCurPhaseInd == -1){
        BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": mCurProbePos: " << mCurProbePos;
        double intensity, variance;
        std::tie(intensity, variance) = measureSpeckleIntensityAndSigma(image, integrationTime);
        nonProbeMeasurmentUpdate(intensity, variance);

    }

    else
        std::tie(mPhaseIntensities[mCurPhaseInd], mPhaseVars[mCurPhaseInd]) = 
                measureSpeckleIntensityAndSigma(image, integrationTime);
         
   if(mCurPhaseInd == NPHASES-1){
       mNProbeIters++;
       updateKalmanState();
       updateNullingSpeckle();
       mCurPhaseInd = -1;

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

dmspeck SpeckleKalman::getNextSpeckle() const{ 
    return mNextSpeck;

}

void SpeckleKalman::nonProbeMeasurmentUpdate(double intensity, double variance){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": rawIntensity: " << intensity;

    if(mParams.get<bool>("KalmanParams.useEKFUpdate") && (mNProbeIters > 0)){
        //mx = mA*mx;
        BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": EKF update";
        cv::Mat H = 2/mDMCalFactor*mx.t(); //Jacobian of I = h(x) = ||x||^2/(alpha)^2
        mP = mP + mQc;
        double S = variance + cv::Mat(H*mP*H.t()).at<double>(0);
        BOOST_LOG_TRIVIAL(debug) << "HPHt: " << cv::Mat(H*mP*H.t());
        BOOST_LOG_TRIVIAL(debug) << " S: " << S;
        mK = mP*H.t()/S;
        double y = intensity - (double)cv::sum(mx.mul(mx))[0];
        mx = mx + mK*y;
        mP = (cv::Mat::eye(mP.rows, mP.cols, CV_64F) - mK*H)*mP;

        BOOST_LOG_TRIVIAL(debug) << "After EKF update: x: \n " << mx;

        cv::Mat amplitude = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
        cv::Mat phase = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);

        cv::Mat real(mx, cv::Range(0, mNProbePos));
        cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
        real = real.reshape(1, mProbeGridWidth);
        imag = imag.reshape(1, mProbeGridWidth);
        cv::cartToPolar(real, imag, amplitude, phase);
        double probeAmp;
        cv::minMaxLoc(amplitude, NULL, &probeAmp, NULL, NULL); 

        mProbeAmp = std::max(probeAmp, mMinProbeAmp);
        
    }

    else{
        mProbeAmp = std::max(mDMCalFactor*sqrt(intensity), mMinProbeAmp);

    }

    mInitialIntensity = intensity;
    mInitialVar = variance;

}
    

void SpeckleKalman::updateKalmanState(){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": updating state estimate";
    int reInd, imInd;
    double varEst;
    std::tie(reInd, imInd) = getKalmanIndices(mCurProbePos);
    mH.setTo(0);
    mH.at<double>(0, reInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    mH.at<double>(1, imInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    mz.at<double>(0) = mPhaseIntensities[0] - mPhaseIntensities[2];
    mz.at<double>(1) = mPhaseIntensities[1] - mPhaseIntensities[3];

    // Relatively unbiased estimate of variance for now, might want to improve later
    varEst = (mPhaseVars[0] + mPhaseVars[1] + mPhaseVars[2] + mPhaseVars[3])/2;
    mR.at<double>(0,0) = std::max(varEst, 
            mH.at<double>(0, reInd)*mx.at<double>(reInd)/2 + 
                mH.at<double>(1, imInd)*mx.at<double>(imInd)/2);
    mR.at<double>(1,1) = mR.at<double>(0, 0); 

    cv::Mat Q = mQ + mQc;

    mx = mA*mx;
    mP = mA*mP*mA.t() + Q;
    cv::Mat S = mR + mH*mP*mH.t();
    mK = mP*mH.t()*S.inv();
    cv::Mat y = mz - mH*mx;
    mx = mx + mK*y;
    mP = (cv::Mat::eye(mP.rows, mP.cols, CV_64F) - mK*mH)*mP;

    BOOST_LOG_TRIVIAL(debug) << "re(x): \n\t" << 
        cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth);
    BOOST_LOG_TRIVIAL(debug) << "im(x): \n\t" << 
        cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth);
    BOOST_LOG_TRIVIAL(debug) << " x_m: \n\t" << 
        mz/(4*mProbeAmp/(mDMCalFactor*mDMCalFactor));
    //BOOST_LOG_TRIVIAL(debug) << "   H: " << mH;
    BOOST_LOG_TRIVIAL(debug) << "   z: \n\t" << mz;
    //BOOST_LOG_TRIVIAL(debug) << "   y: " << y;
    //BOOST_LOG_TRIVIAL(debug) << "   Ky: " << mK*y;
    BOOST_LOG_TRIVIAL(debug) << "   R: \n\t" << mR;
    BOOST_LOG_TRIVIAL(debug) << "   Q: \n\t" << Q;
    //BOOST_LOG_TRIVIAL(debug) << "   K: " << mK;
    //BOOST_LOG_TRIVIAL(debug) << "   P: " << mP;
    
    mQc.setTo(0);
    
    BOOST_LOG_TRIVIAL(debug) << "N probe iters: " << mNProbeIters;

    std::tie(mCurProbePos.x, mCurProbePos.y) = getProbeGridIndices((int)std::rand()%mNProbePos);

}

void SpeckleKalman::updateNullingSpeckle(){
    cv::Mat amplitude = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::Mat phase = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": calculating nulling speckle";


    cv::Mat real(mx, cv::Range(0, mNProbePos));
    cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
    real = real.reshape(1, mProbeGridWidth);
    imag = imag.reshape(1, mProbeGridWidth);
    
    cv::Mat variance = mP.diag();
    variance = cv::Mat(variance, cv::Range(0, mNProbePos)) + cv::Mat(variance, cv::Range(mNProbePos, 2*mNProbePos));
    variance = variance.reshape(1, mProbeGridWidth);
    assert(variance.cols == mProbeGridWidth);
    
    cv::cartToPolar(real, imag, amplitude, phase);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Amplitudes: " << amplitude;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Phases: " << phase;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Variances: " << variance;

    // weights_ij is proportional to amplitude_ij/(overlap of probe grid w/ gaussian)
    cv::Mat weights = cv::Mat::ones(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::GaussianBlur(weights, weights, cv::Size(mProbeGridWidth, mProbeGridWidth), 4*M_PI*0.42, 0, cv::BORDER_CONSTANT);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: normweights: " << weights;
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
            //nullingAmp += weights.at<double>(position[0], position[1])*amplitude.at<double>(position[0], position[1]);
            nullingPhase += weights.at<double>(position[0], position[1])*phase.at<double>(position[0], position[1]); 

            });

    //nullingVar = cv::sum(variance.mul(weights))[0]; //this could also be done inside lambda function
    //Use the maximum amplitude as nulling amp, as well as the variance at this value
    cv::Point2i nullingAmpLoc;
    cv::minMaxLoc(amplitude, NULL, &nullingAmp, NULL, &nullingAmpLoc); 
    nullingVar = variance.template at<double>(nullingAmpLoc);

    double snr = nullingAmp/std::sqrt(nullingVar);

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling k: " << nullingK;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling amplitude: " << nullingAmp;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling phase: " << nullingPhase;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling variance: " << nullingVar;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling snr: " << snr;

    mNextSpeck.kx = nullingK.x;
    mNextSpeck.ky = nullingK.y;
    mNextSpeck.phase = nullingPhase + M_PI;
    mNextSpeck.isNull = true; 

    if((snr >= mParams.get<double>("KalmanParams.snrThresh")) && (mNProbeIters >= mMinProbeIters)){
        mNextSpeck.amp = nullingAmp; //nullingAmp;

        BOOST_LOG_TRIVIAL(info) << "SpeckleKalman at " << mCoords << ": applying nulling speckle after "
            << mNProbeIters << " probe iterations. \n\t k: " << nullingK << " amp: " << nullingAmp;

        //Update state est w/ control
        cv::Mat B(2*mNProbePos, 2, CV_64F, cv::Scalar(0));
        cv::Mat realB(B, cv::Range(0, mNProbePos), cv::Range(0,1));
        cv::Mat imagB(B, cv::Range(mNProbePos, 2*mNProbePos), cv::Range(1,2));
        cv::Mat realQc(mQc, cv::Range(0, mNProbePos), cv::Range(0, mNProbePos));
        cv::Mat imagQc(mQc, cv::Range(mNProbePos, 2*mNProbePos), cv::Range(mNProbePos, 2*mNProbePos));
        int kRow, kCol;
        double kDist;
        double posCorr;
        for(int i=0; i<mNProbePos; i++){
            std::tie(kRow, kCol) = getProbeGridIndices(i);
            kDist = cv::norm(cv::Point2d(mNextSpeck.kx, mNextSpeck.ky) - mProbeGridKvecs.at<cv::Point2d>(kRow, kCol));
            posCorr = std::exp(-kDist*kDist/(4*mKvecCorrSigma*mKvecCorrSigma));

            realB.at<double>(i) = posCorr; 
            imagB.at<double>(i) = posCorr; 

            realQc.at<double>(i, i) = 4*mx.at<double>(i)*mParams.get<double>("KalmanParams.calVar")/(mDMCalFactor*mDMCalFactor)*posCorr;
            imagQc.at<double>(i, i) = 4*mx.at<double>(i + mNProbePos)*mParams.get<double>("KalmanParams.calVar")/(mDMCalFactor*mDMCalFactor)*posCorr;

        }

        correlateProcessNoise(mQc);

        //BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: B: " << B;

        cv::Mat u(2, 1, CV_64F, cv::Scalar(0));
        u.at<double>(0) = mNextSpeck.amp*std::cos(mNextSpeck.phase);
        u.at<double>(1) = mNextSpeck.amp*std::sin(mNextSpeck.phase);
        mx = mx + B*u;
        BOOST_LOG_TRIVIAL(info) << "SpeckleKalman at " << mCoords << ": x after null:" << mx;
        
    }

    else
        mNextSpeck.amp = 0;
    
    
}
    

void SpeckleKalman::correlateProcessNoise(cv::Mat &noiseMat){
    cv::Mat realBlock = cv::Mat(noiseMat, cv::Range(0, mNProbePos), cv::Range(0, mNProbePos));
    cv::Mat imagBlock = cv::Mat(noiseMat, cv::Range(mNProbePos, 2*mNProbePos), cv::Range(mNProbePos, 2*mNProbePos));
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
            
            realBlock.at<double>(i, j) = std::sqrt(realBlock.at<double>(i, i)*realBlock.at<double>(j,j))*std::exp(-kDist*kDist/(4*mKvecCorrSigma*mKvecCorrSigma));
            imagBlock.at<double>(i, j) = std::sqrt(imagBlock.at<double>(i,i)*imagBlock.at<double>(j,j))*std::exp(-kDist*kDist/(4*mKvecCorrSigma*mKvecCorrSigma));

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

