#include "SpeckleKalman.h"

SpeckleKalman::SpeckleKalman(cv::Point2d pt, boost::property_tree::ptree &ptree):
        SpeckleController(pt, ptree){
    for(int i=0; i<NPHASES; i++)
        mPhaseList[i] = (double)2*M_PI*i/NPHASES;
     
    mCVFormatter = cv::Formatter::get();
    mCVFormatter->set32fPrecision(2);
    mCVFormatter->set64fPrecision(2);

    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mProbeGridSpacing = mParams.get<double>("KalmanParams.probeGridSpacing");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;
    mKvecCorrSigma = 0.42*2*M_PI;
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
    mProbeGridCounter = cv::Mat(mProbeGridWidth, mProbeGridWidth, CV_64F, cv::Scalar(0));
    mNullingGain = mParams.get<double>("KalmanParams.nullingGain");



}

dmspeck SpeckleKalman::getNextProbeSpeckle(int phaseInd){
   cv::Point2d curKvecs = mProbeGridKvecs.at<cv::Point2d>(mCurProbePos);
   dmspeck nextSpeck;
   nextSpeck.kx = curKvecs.x;
   nextSpeck.ky = curKvecs.y;
   nextSpeck.amp = mProbeAmp;
   nextSpeck.phase = mPhaseList[phaseInd];
   nextSpeck.isNull = false;
   return nextSpeck;

}

void SpeckleKalman::probeMeasurementUpdate(int phaseInd, double intensity, double variance){
   mPhaseIntensities[phaseInd] = intensity;
   mPhaseVars[phaseInd] = variance;

}
    


void SpeckleKalman::nonProbeMeasurementUpdate(double intensity, double variance){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": rawIntensity: " << intensity;

    if(mParams.get<bool>("KalmanParams.useEKFUpdate") && (getNProbeIters() > 0)){
        //mx = mA*mx;
        double measVarEst;
        //cv::minMaxLoc(mx, NULL, &measVarEst, NULL, NULL);
        measVarEst = cv::norm(mx)*4*mProbeAmp/(mDMCalFactor*mDMCalFactor)*1000/mLastIntTime;
        variance = std::max(variance, measVarEst);
        BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": EKF update";
        cv::Mat H = 2/(mDMCalFactor*mDMCalFactor*mNProbePos)*mx.t(); //Jacobian of I = h(x) = ||x||^2/(alpha)^2
        mP = mP + mQc;
        double S = variance + cv::Mat(H*mP*H.t()).at<double>(0);
        BOOST_LOG_TRIVIAL(debug) << "H: " << mCVFormatter->format(H);
        BOOST_LOG_TRIVIAL(debug) << "variance: " << variance;
        BOOST_LOG_TRIVIAL(debug) << "measVarEst: " << measVarEst;
        BOOST_LOG_TRIVIAL(trace) << "HPHt: " << mCVFormatter->format(cv::Mat(H*mP*H.t()));
        BOOST_LOG_TRIVIAL(trace) << " S: " << S;
        //mK = mP*H.t()/S;
        //mK = mP.diag().mul(H.t())/S;
        mK = cv::mean(mP.diag())[0]*H.t()/S;
        double y = intensity - (double)cv::sum(mx.mul(mx))[0]/(mDMCalFactor*mDMCalFactor*mNProbePos);
        mx = mx + mK*y;
        mP = (cv::Mat::eye(mP.rows, mP.cols, CV_64F) - mK*H)*mP;

        BOOST_LOG_TRIVIAL(debug) << "After EKF update: x: \n " << mCVFormatter->format(mx);
        BOOST_LOG_TRIVIAL(debug) << "                  y: \n " << y;
        BOOST_LOG_TRIVIAL(debug) << "                  K: \n " << mCVFormatter->format(mK);
        BOOST_LOG_TRIVIAL(debug) << "                 Ky: \n " << mCVFormatter->format(mK*y);
        BOOST_LOG_TRIVIAL(trace) << "                  P: \n " << mCVFormatter->format(mP);

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

        mQc.setTo(0);
        
    }

    else{
        cv::Mat amplitude = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
        cv::Mat phase = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);

        cv::Mat real(mx, cv::Range(0, mNProbePos));
        cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
        real = real.reshape(1, mProbeGridWidth);
        imag = imag.reshape(1, mProbeGridWidth);
        cv::cartToPolar(real, imag, amplitude, phase);
        double probeAmp;
        cv::minMaxLoc(amplitude, NULL, &probeAmp, NULL, NULL); 

        //probeAmp = std::max(probeAmp, std::sqrt(intensity)*mDMCalFactor);
        if(getNProbeIters()==0)
            mProbeAmp = std::sqrt(intensity)*mDMCalFactor;
        else
            mProbeAmp = std::max(probeAmp, mMinProbeAmp);

    }

    if(getNProbeIters()%10 == 0)
        BOOST_LOG_TRIVIAL(info) << "Iter: " << getNProbeIters() << "; Probing at: " << mProbeAmp;


}

dmspeck SpeckleKalman::endOfProbeUpdate(){
    updateKalmanState();
    return updateNullingSpeckle();

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

    BOOST_LOG_TRIVIAL(debug) << "mCurProbePos: " << mCurProbePos;
    BOOST_LOG_TRIVIAL(debug) << "re(x): \n" << 
        mCVFormatter->format(cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth));
    BOOST_LOG_TRIVIAL(debug) << "im(x): \n" << 
        mCVFormatter->format(cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth));
    BOOST_LOG_TRIVIAL(debug) << " x_m: \n" << 
        mCVFormatter->format(mz/(4*mProbeAmp/(mDMCalFactor*mDMCalFactor)));
    //BOOST_LOG_TRIVIAL(debug) << "   H: " << mH;
    BOOST_LOG_TRIVIAL(trace) << "   z: \n" << mCVFormatter->format(mz);
    //BOOST_LOG_TRIVIAL(debug) << "   y: " << y;
    //BOOST_LOG_TRIVIAL(debug) << "   Ky: " << mK*y;
    BOOST_LOG_TRIVIAL(trace) << "   R: \n" << mCVFormatter->format(mR);
    BOOST_LOG_TRIVIAL(trace) << "   Q: \n" << mCVFormatter->format(Q);
    //BOOST_LOG_TRIVIAL(debug) << "   K: " << mK;
    BOOST_LOG_TRIVIAL(trace) << "   P: " << mCVFormatter->format(mP);
    
    mQc.setTo(0);
    
    BOOST_LOG_TRIVIAL(debug) << "N probe iters: " << getNProbeIters();

    updateProbeGridIndices();


}

void SpeckleKalman::updateProbeGridIndices(){
    mProbeGridCounter.at<double>(mCurProbePos.x, mCurProbePos.y) += 1;
    //std::tie(mCurProbePos.x, mCurProbePos.y) = getProbeGridIndices((int)std::rand()%mNProbePos);
    int reInd, imInd;
    std::tie(reInd, imInd) = getKalmanIndices(mCurProbePos);
    //std::cout << "reind: " << reInd;
    std::tie(mCurProbePos.y, mCurProbePos.x) = getProbeGridIndices((reInd + 1)%mNProbePos);

}


dmspeck SpeckleKalman::updateNullingSpeckle(){
    cv::Mat amplitudeGrid = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::Mat phaseGrid = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman at " << mCoords << ": calculating nulling speckle";


    cv::Mat real(mx, cv::Range(0, mNProbePos));
    cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
    real = real.reshape(1, mProbeGridWidth);
    imag = imag.reshape(1, mProbeGridWidth);
    
    cv::Mat variance = mP.diag();
    variance = cv::Mat(variance, cv::Range(0, mNProbePos)) + cv::Mat(variance, cv::Range(mNProbePos, 2*mNProbePos));
    variance = variance.reshape(1, mProbeGridWidth);
    assert(variance.cols == mProbeGridWidth);
    
    cv::cartToPolar(real, imag, amplitudeGrid, phaseGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Amplitudes: \n" << mCVFormatter->format(amplitudeGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Phases: \n" << mCVFormatter->format(phaseGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Variances: \n" << mCVFormatter->format(variance);

    // weights_ij is proportional to amplitudeGrid_ij/(overlap of probe grid w/ gaussian)
    cv::Mat weights = cv::Mat::ones(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::GaussianBlur(weights, weights, cv::Size(mProbeGridWidth, mProbeGridWidth), 4*M_PI*0.42, 0, cv::BORDER_CONSTANT);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: normweights: \n" << mCVFormatter->format(weights);
    cv::divide(amplitudeGrid, weights, weights);
    weights = weights.mul(weights);
    weights = weights/cv::sum(weights)[0];    

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: weights \n" << mCVFormatter->format(weights);

    cv::Point2d nullingK;
    double nullingAmp;
    double nullingPhase;
    double nullingVar;
    double realSum;
    double imagSum;
    mProbeGridKvecs.forEach([this, &weights, &nullingK, &nullingAmp, &nullingPhase]
                (cv::Point2d &value, const int *position) -> void{
            nullingK += weights.at<double>(position[0], position[1])*value;
            //nullingAmp += weights.at<double>(position[0], position[1])*amplitudeGrid.at<double>(position[0], position[1]);
            //nullingPhase += weights.at<double>(position[0], position[1])*phaseGrid.at<double>(position[0], position[1]); 

            });

    //nullingVar = cv::sum(variance.mul(weights))[0]; //this could also be done inside lambda function
    //Use the maximum amplitudeGrid as nulling amp, as well as the variance at this value
    realSum = (double)cv::sum(real.mul(weights))[0];
    imagSum = (double)cv::sum(imag.mul(weights))[0];
    nullingPhase = std::atan2(imagSum, realSum);
    cv::Point2i nullingAmpLoc;
    cv::minMaxLoc(amplitudeGrid, NULL, &nullingAmp, NULL, &nullingAmpLoc); 
    nullingVar = variance.at<double>(nullingAmpLoc);
    nullingAmp *= mNullingGain;

    double snr = nullingAmp/(mNullingGain*std::sqrt(nullingVar));

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling k: " << nullingK;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling amplitude: " << nullingAmp;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling phase: " << nullingPhase;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling variance: " << nullingVar;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalman: Nulling snr: " << snr;

    dmspeck nullingSpeck;

    nullingSpeck.kx = nullingK.x;
    nullingSpeck.ky = nullingK.y;
    nullingSpeck.phase = nullingPhase + M_PI;
    nullingSpeck.isNull = true; 

    if((snr >= mParams.get<double>("KalmanParams.snrThresh")) && (getNProbeIters() >= mMinProbeIters)){
        nullingSpeck.amp = nullingAmp; //nullingAmp;

        BOOST_LOG_TRIVIAL(info) << "SpeckleKalman at " << mCoords << ": applying nulling speckle after "
            << getNProbeIters() << " probe iterations. \n\t k: " << nullingK << " amp: " << nullingAmp << "phase: " << nullingPhase;
        BOOST_LOG_TRIVIAL(info) << "weights: \n" << weights;
        BOOST_LOG_TRIVIAL(info) << "    mx pre-null: ";
        BOOST_LOG_TRIVIAL(info) << "       re: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth));
        BOOST_LOG_TRIVIAL(info) << "       im: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth));

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
            kDist = cv::norm(cv::Point2d(nullingSpeck.kx, nullingSpeck.ky) - mProbeGridKvecs.at<cv::Point2d>(kRow, kCol));
            posCorr = std::exp(-kDist*kDist/(4*mKvecCorrSigma*mKvecCorrSigma));

            realB.at<double>(i) = posCorr; 
            imagB.at<double>(i) = posCorr; 

            realQc.at<double>(i, i) = 4*mNullingGain*std::pow(mx.at<double>(i), 2)*std::pow(mParams.get<double>("KalmanParams.calSigma"), 2)/(mDMCalFactor*mDMCalFactor)*posCorr;
            imagQc.at<double>(i, i) = 4*mNullingGain*std::pow(mx.at<double>(i + mNProbePos), 2)*std::pow(mParams.get<double>("KalmanParams.calSigma"), 2)/(mDMCalFactor*mDMCalFactor)*posCorr;

        }

        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalman: precorr mQc: " << mCVFormatter->format(mQc);

        correlateProcessNoise(mQc);

        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalman: mQc: " << mCVFormatter->format(mQc);
        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalman: B: " << mCVFormatter->format(B);

        cv::Mat u(2, 1, CV_64F, cv::Scalar(0));
        u.at<double>(0) = nullingSpeck.amp*std::cos(nullingSpeck.phase);
        u.at<double>(1) = nullingSpeck.amp*std::sin(nullingSpeck.phase);
        mx = mx + B*u;
        BOOST_LOG_TRIVIAL(info) << "    mx post-null: ";
        BOOST_LOG_TRIVIAL(info) << "       re: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth));
        BOOST_LOG_TRIVIAL(info) << "       im: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth));
        BOOST_LOG_TRIVIAL(info) << "SpeckleKalman: Probe Grid Counter: \n" << mProbeGridCounter;

    }

    else
        nullingSpeck.amp = 0;

    return nullingSpeck;
    
    
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

