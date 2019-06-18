#include "SpeckleKalmanPoisson.h"

SpeckleKalmanPoisson::SpeckleKalmanPoisson(cv::Point2d pt, boost::property_tree::ptree &ptree):
        SpeckleController(pt, ptree){
    for(int i=0; i<NPHASES; i++)
        mPhaseList[i] = (double)2*M_PI*i/NPHASES;
     
    mCVFormatter = cv::Formatter::get(cv::Formatter::FMT_PYTHON);
    mCVFormatter->set32fPrecision(2);
    mCVFormatter->set64fPrecision(2);

    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mProbeGridSpacing = mParams.get<double>("KalmanParams.probeGridSpacing");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;
    mKvecCorrSigma = 0.42*2*M_PI;
    mCurProbePos = cv::Point2i(mProbeGridWidth/2, mProbeGridWidth/2);
    mDMCalFactor = getDMCalFactorCPS(mKvecs, mParams.get<double>("DMParams.a"), mParams.get<double>("DMParams.b"), 
            mParams.get<double>("DMParams.c"));
    mDMCalSigma = mParams.get<double>("KalmanParams.calSigma")/(2*mDMCalFactor);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: DMCalFactor: " << mDMCalFactor;

    mx = cv::Mat::zeros(2*mNProbePos, 1, CV_64F);
    mA = cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mPi = mParams.get<double>("KalmanParams.initStateVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQ = mParams.get<double>("KalmanParams.processNoiseVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQc = cv::Mat::zeros(2*mNProbePos, 2*mNProbePos, CV_64F);
    mR = cv::Mat::zeros(2, 2, CV_64F);
    mMinProbeAmp = 2;

    initializeProbeGridKvecs();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Probe Grid: " << mProbeGridKvecs;

    correlateProcessNoise(mPi);
    //BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Initial State Variance: " << mP;

    mMinProbeIters = mParams.get<int>("KalmanParams.minProbeIters");
    mProbeGridCounter = cv::Mat(mProbeGridWidth, mProbeGridWidth, CV_64F, cv::Scalar(0));
    mNullingGain = mParams.get<double>("KalmanParams.nullingGain");



}

dmspeck SpeckleKalmanPoisson::getNextProbeSpeckle(int phaseInd){
   cv::Point2d curKvecs = mProbeGridKvecs.at<cv::Point2d>(mCurProbePos);
   dmspeck nextSpeck;
   nextSpeck.kx = curKvecs.x;
   nextSpeck.ky = curKvecs.y;
   nextSpeck.amp = mProbeAmp;
   nextSpeck.phase = mPhaseList[phaseInd];
   nextSpeck.isNull = false;
   return nextSpeck;

}

void SpeckleKalmanPoisson::probeMeasurementUpdate(int phaseInd, double intensity, double variance){
   mPhaseIntensities[phaseInd] = intensity;
   mPhaseVars[phaseInd] = variance;

}
    


void SpeckleKalmanPoisson::nonProbeMeasurementUpdate(double intensity, double variance){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson at " << mCoords << ": rawIntensity: " << intensity;

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
    //else
    //    mProbeAmp = std::max(probeAmp, mMinProbeAmp);

    

    if(getNProbeIters()%10 == 0)
        BOOST_LOG_TRIVIAL(info) << "Iter: " << getNProbeIters() << "; Probing at: " << mProbeAmp;


}

dmspeck SpeckleKalmanPoisson::endOfProbeUpdate(){
    updateKalmanState();
    return updateNullingSpeckle();

}
    
    

void SpeckleKalmanPoisson::updateKalmanState(){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson at " << mCoords << ": updating state estimate";
    int reInd, imInd;
    double varEst;
    std::tie(reInd, imInd) = getKalmanIndices(mCurProbePos);
    cv::Mat H = cv::Mat::zeros(2, 2*mNProbePos, CV_64F);
    cv::Mat z = cv::Mat::zeros(2, 1, CV_64F);
    H.at<double>(0, reInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    H.at<double>(1, imInd) = 4*mProbeAmp/(mDMCalFactor*mDMCalFactor);
    z.at<double>(0) = mPhaseIntensities[0] - mPhaseIntensities[2];
    z.at<double>(1) = mPhaseIntensities[1] - mPhaseIntensities[3];
    mHList.push_back(H);
    mzList.push_back(z);

    // running average of measurements to "unbias" R
    varEst = (mPhaseVars[0] + mPhaseVars[1] + mPhaseVars[2] + mPhaseVars[3])/2;
    varEst = std::max(varEst, 
                    H.at<double>(0, reInd)*mx.at<double>(reInd)/2 + 
                        H.at<double>(1, imInd)*mx.at<double>(imInd)/2);

    mR.at<double>(0,0) = (mzList.size() - 1)*mR.at<double>(0,0)/mzList.size() + varEst/mzList.size();
    mR.at<double>(1,1) = mR.at<double>(0, 0); 

    mP = mPi.clone();
    mx.setTo(0);
    cv::Mat S, K, y;
    BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: propagating filter...";
    for(int i=0; i<mzList.size(); i++){
        mx = mA*mx;
        mP = mA*mP*mA.t() + mQ;
        S = mR + mHList[i]*mP*mHList[i].t();
        K = mP*mHList[i].t()*S.inv();
        y = mzList[i] - mHList[i]*mx;
        mx = mx + K*y;
        mP = (cv::Mat::eye(mP.rows, mP.cols, CV_64F) - K*mHList[i])*mP;
        if(mParams.get<bool>("KalmanParams.forceCovariance"))
            correlateProcessNoise(mP);
        BOOST_LOG_TRIVIAL(trace) << "     z[" << i << "]: \n" << mCVFormatter->format(mzList[i]);
        BOOST_LOG_TRIVIAL(trace) << "     H[" << i << "]: \n" << mCVFormatter->format(mHList[i]);
        BOOST_LOG_TRIVIAL(trace) << "     S[" << i << "]: \n" << mCVFormatter->format(S);
        BOOST_LOG_TRIVIAL(trace) << "     K[" << i << "]: \n" << mCVFormatter->format(K);
        BOOST_LOG_TRIVIAL(trace) << "  I-KH[" << i << "]: \n" << mCVFormatter->format(cv::Mat::eye(mP.rows, mP.cols, CV_64F) 
                - K*mHList[i]);
        BOOST_LOG_TRIVIAL(trace) << "    mP[" << i << "]: \n" << mCVFormatter->format(mP);
        BOOST_LOG_TRIVIAL(trace) << "    mx[" << i << "]: \n" << mCVFormatter->format(mx);

    }

    BOOST_LOG_TRIVIAL(debug) << "mCurProbePos: " << mCurProbePos;
    BOOST_LOG_TRIVIAL(debug) << "re(x): \n" << 
        mCVFormatter->format(cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth));
    BOOST_LOG_TRIVIAL(debug) << "im(x): \n" << 
        mCVFormatter->format(cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth));
    BOOST_LOG_TRIVIAL(debug) << " x_m: \n" << 
        mCVFormatter->format(z/(4*mProbeAmp/(mDMCalFactor*mDMCalFactor)));
    //BOOST_LOG_TRIVIAL(debug) << "   H: " << mH;
    BOOST_LOG_TRIVIAL(trace) << "   z: \n" << mCVFormatter->format(z);
    //BOOST_LOG_TRIVIAL(debug) << "   y: " << y;
    //BOOST_LOG_TRIVIAL(debug) << "   Ky: " << mK*y;
    BOOST_LOG_TRIVIAL(trace) << "   R: \n" << mCVFormatter->format(mR);
    BOOST_LOG_TRIVIAL(trace) << "   Q: \n" << mCVFormatter->format(mQ);
    //BOOST_LOG_TRIVIAL(debug) << "   K: " << mK;
    BOOST_LOG_TRIVIAL(trace) << "   P: " << mCVFormatter->format(mP);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson varEst: " << mR.at<double>(0, 0) << " after " << mHList.size();
    
    mQc.setTo(0);
    
    BOOST_LOG_TRIVIAL(debug) << "N probe iters: " << getNProbeIters();

    updateProbeGridIndices();


}

void SpeckleKalmanPoisson::updateProbeGridIndices(){
    mProbeGridCounter.at<double>(mCurProbePos.x, mCurProbePos.y) += 1;
    //std::tie(mCurProbePos.x, mCurProbePos.y) = getProbeGridIndices((int)std::rand()%mNProbePos);
    int reInd, imInd;
    std::tie(reInd, imInd) = getKalmanIndices(mCurProbePos);
    //std::cout << "reind: " << reInd;
    std::tie(mCurProbePos.y, mCurProbePos.x) = getProbeGridIndices((reInd + 1)%mNProbePos);

}


dmspeck SpeckleKalmanPoisson::updateNullingSpeckle(){
    cv::Mat amplitudeGrid = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    cv::Mat phaseGrid = cv::Mat::zeros(mProbeGridWidth, mProbeGridWidth, CV_64F);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson at " << mCoords << ": calculating nulling speckle";


    cv::Mat real(mx, cv::Range(0, mNProbePos));
    cv::Mat imag(mx, cv::Range(mNProbePos, 2*mNProbePos));
    real = real.reshape(1, mProbeGridWidth);
    imag = imag.reshape(1, mProbeGridWidth);
    
    cv::Mat variance = mP.diag();
    variance = 0.5*cv::Mat(variance, cv::Range(0, mNProbePos)) + 0.5*cv::Mat(variance, cv::Range(mNProbePos, 2*mNProbePos));
    variance = variance.reshape(1, mProbeGridWidth);
    assert(variance.cols == mProbeGridWidth);
    
    cv::cartToPolar(real, imag, amplitudeGrid, phaseGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Amplitudes: \n" << mCVFormatter->format(amplitudeGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Phases: \n" << mCVFormatter->format(phaseGrid);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Variances: \n" << mCVFormatter->format(variance);

    // weights_ij is proportional to amplitudeGrid_ij/(overlap of probe grid w/ gaussian)
    int wUSFactor = 2;
    double resizeFactor = (double)(mProbeGridWidth*2 - 1)/(mProbeGridWidth);
    int usSize = (int)std::round(resizeFactor*mProbeGridWidth);
    cv::Mat overlapGrid = cv::Mat::ones(usSize, usSize, CV_64F);
    cv::GaussianBlur(overlapGrid, overlapGrid, cv::Size(2*usSize + 1,2*usSize + 1), wUSFactor*mKvecCorrSigma/mProbeGridSpacing, 0, cv::BORDER_CONSTANT);

    cv::Mat weights = cv::Mat::zeros(usSize, usSize, CV_64F);
    cv::Mat amplitudeGridUS, realUS, imagUS;
    cv::resize(amplitudeGrid, amplitudeGridUS, cv::Size(0,0), resizeFactor, resizeFactor);
    cv::resize(real, realUS, cv::Size(0,0), resizeFactor, resizeFactor);
    cv::resize(imag, imagUS, cv::Size(0,0), resizeFactor, resizeFactor);
    cv::GaussianBlur(amplitudeGridUS, weights, cv::Size(usSize, usSize), 0.25*wUSFactor*mKvecCorrSigma/mProbeGridSpacing, 0, cv::BORDER_REFLECT_101); //, cv::BORDER_CONSTANT);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: amplitudeGridConv \n" << mCVFormatter->format(weights);
    //BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: normweights: \n" << mCVFormatter->format(weights);
    //cv::divide(weights, overlapGrid, weights); // divide weights by overlap fraction with kernel
    //weights = weights.mul(weights);
    weights = weights/cv::sum(weights)[0];    

    mCVFormatter->set64fPrecision(4);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: weights \n" << mCVFormatter->format(weights);
    mCVFormatter->set64fPrecision(2);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: amplitudeGridUS \n" << mCVFormatter->format(amplitudeGridUS);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: overlapGrid \n" << mCVFormatter->format(overlapGrid);
    cv::Mat kernel = cv::Mat::zeros(usSize, usSize, CV_64F);
    kernel.at<double>(usSize/2, usSize/2) = 1;
    cv::GaussianBlur(kernel, kernel, cv::Size(usSize, usSize), 0.25*wUSFactor*mKvecCorrSigma/mProbeGridSpacing, 0, cv::BORDER_CONSTANT);
    mCVFormatter->set64fPrecision(4);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: kernel \n" << mCVFormatter->format(kernel);
    mCVFormatter->set64fPrecision(2);


    cv::Point2d nullingK;
    cv::Point2i nullingAmpLoc;
    double nullingAmp;
    double nullingPhase;
    double nullingVar;
    double realSum;
    double imagSum;
    //mProbeGridKvecs.forEach([this, &weights, &nullingK, &nullingAmp, &nullingPhase]
    //            (cv::Point2d &value, const int *position) -> void{
    //        nullingK += weights.at<double>(position[0], position[1])*value;
    //        //nullingAmp += weights.at<double>(position[0], position[1])*amplitudeGrid.at<double>(position[0], position[1]);
    //        //nullingPhase += weights.at<double>(position[0], position[1])*phaseGrid.at<double>(position[0], position[1]); 

    //        });

    //nullingVar = cv::sum(variance.mul(weights))[0]; //this could also be done inside lambda function
    //Use the maximum amplitudeGrid as nulling amp, as well as the variance at this value
    realSum = (double)cv::sum(realUS.mul(weights))[0];
    imagSum = (double)cv::sum(imagUS.mul(weights))[0];
    nullingPhase = std::atan2(imagSum, realSum);
    cv::minMaxLoc(weights, NULL, NULL, NULL, &nullingAmpLoc); 
    nullingAmp = amplitudeGridUS.at<double>(nullingAmpLoc);
    nullingK.x = mKvecs.x + mProbeGridSpacing*((double)nullingAmpLoc.x/wUSFactor - (int)mProbeGridWidth/2);
    nullingK.y = mKvecs.y + mProbeGridSpacing*((double)nullingAmpLoc.y/wUSFactor - (int)mProbeGridWidth/2);
    nullingVar = variance.at<double>(nullingAmpLoc/2);
    nullingAmp *= mNullingGain;

    double snr = nullingAmp/(mNullingGain*std::sqrt(nullingVar));

    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Nulling k: " << nullingK;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Nulling amplitude: " << nullingAmp;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Nulling phase: " << nullingPhase;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Nulling variance: " << nullingVar;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleKalmanPoisson: Nulling snr: " << snr;

    dmspeck nullingSpeck;

    nullingSpeck.kx = nullingK.x;
    nullingSpeck.ky = nullingK.y;
    nullingSpeck.phase = nullingPhase + M_PI;
    nullingSpeck.isNull = true; 

    if((snr >= mParams.get<double>("KalmanParams.snrThresh")) && (getNProbeIters() >= mMinProbeIters)){
        nullingSpeck.amp = nullingAmp; //nullingAmp;

        BOOST_LOG_TRIVIAL(info) << "SpeckleKalmanPoisson at " << mCoords << ": applying nulling speckle after "
            << getNProbeIters() << " probe iterations. \n\t k: " << nullingK << " amp: " << nullingAmp << " phase: " << nullingPhase;
        mCVFormatter->set64fPrecision(4);
        BOOST_LOG_TRIVIAL(info) << "    weights: \n" << mCVFormatter->format(weights);
        BOOST_LOG_TRIVIAL(info) << "    nullingloc: \n" << nullingAmpLoc;
        mCVFormatter->set64fPrecision(2);
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
            BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: posCorr mQc: " << posCorr;
            BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: mx mQc: " << mx.at<double>(i);
            BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: mDMCalFactor mQc: " << mDMCalFactor;

            realB.at<double>(i) = posCorr; 
            imagB.at<double>(i) = posCorr; 

            realQc.at<double>(i, i) = 4*mNullingGain*std::pow(mx.at<double>(i), 2)*mDMCalSigma*mDMCalSigma/(mDMCalFactor*mDMCalFactor)*posCorr;
            imagQc.at<double>(i, i) = 4*mNullingGain*std::pow(mx.at<double>(i + mNProbePos), 2)*mDMCalSigma*mDMCalSigma/(mDMCalFactor*mDMCalFactor)*posCorr;

        }

        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: precorr mQc: " << mCVFormatter->format(mQc);

        correlateProcessNoise(mQc);

        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: mQc: " << mCVFormatter->format(mQc);
        BOOST_LOG_TRIVIAL(trace) << "SpeckleKalmanPoisson: B: " << mCVFormatter->format(B);

        cv::Mat u(2, 1, CV_64F, cv::Scalar(0));
        u.at<double>(0) = nullingSpeck.amp*std::cos(nullingSpeck.phase);
        u.at<double>(1) = nullingSpeck.amp*std::sin(nullingSpeck.phase);
        mx = mx + B*u;
        BOOST_LOG_TRIVIAL(info) << "    mx post-null: ";
        BOOST_LOG_TRIVIAL(info) << "       re: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(0, mNProbePos)).reshape(0, mProbeGridWidth));
        BOOST_LOG_TRIVIAL(info) << "       im: \n" << 
            mCVFormatter->format(cv::Mat(mx, cv::Range(mNProbePos, 2*mNProbePos)).reshape(0, mProbeGridWidth));
        BOOST_LOG_TRIVIAL(info) << "SpeckleKalmanPoisson: Probe Grid Counter: \n" << mProbeGridCounter;

        //Update Covariance 
        mP = mP + mQc;
        mPi = mP.clone();
        mzList.clear();
        mHList.clear();

        //Update Probe amplitude
        mProbeAmp = std::max(mMinProbeAmp, (1-mNullingGain)*mProbeAmp);

    }

    else
        nullingSpeck.amp = 0;

    return nullingSpeck;
    
    
}
    

void SpeckleKalmanPoisson::correlateProcessNoise(cv::Mat &noiseMat){
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

void SpeckleKalmanPoisson::initializeProbeGridKvecs()
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

std::tuple<int, int> SpeckleKalmanPoisson::getKalmanIndices(int r, int c){
    int realInd = r*mProbeGridWidth + c;
    int imagInd = realInd + mNProbePos;
    return std::make_tuple(realInd, imagInd);

}

std::tuple<int, int> SpeckleKalmanPoisson::getKalmanIndices(cv::Point2i &probePos){
    int realInd = probePos.y*mProbeGridWidth + probePos.x;
    int imagInd = realInd + mNProbePos;
    return std::make_tuple(realInd, imagInd);

}

std::tuple<int, int> SpeckleKalmanPoisson::getProbeGridIndices(int kalmanInd)
{
    int row = kalmanInd/mProbeGridWidth;
    int col = kalmanInd%mProbeGridWidth;
    return std::make_tuple(row, col);

}

