#include "SpeckleBasic.h"

SpeckleBasic::SpeckleBasic(cv::Point2d pt, boost::property_tree::ptree &cfgParams): SpeckleController(pt, cfgParams){
    for(int i=0; i<NPHASES; i++){
        mPhaseIntensities[i] = 0;
        mPhaseVars[i] = 0;

    }

    mDMCalFactor = getDMCalFactorCPS(mKvecs, mParams.get<double>("DMParams.a"), mParams.get<double>("DMParams.b"), 
            mParams.get<double>("DMParams.c"));

}
        

void SpeckleBasic::update(const cv::Mat &image, double integrationTime){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": mCurPhaseInd: " << mCurPhaseInd;
    if(mCurPhaseInd == -1){
        double intensity, variance;
        std::tie(intensity, variance) = measureSpeckleIntensityAndSigma(image, integrationTime);
        nonProbeMeasurmentUpdate(intensity, variance);

    }

    else{
        double intensity, variance;
        std::tie(intensity, variance) = measureSpeckleIntensityAndSigma(image, integrationTime);
        mPhaseIntensities[mCurPhaseInd] += intensity;
        mPhaseVars[mCurPhaseInd] += variance;

    }
         
   if(mCurPhaseInd == NPHASES-1){
       mNProbeIters++;
       updateNullingSpeckle();
       mCurPhaseInd = -1;

   }

   else{
       mCurPhaseInd += 1;
       mNextSpeck.kx = mKvecs.x;
       mNextSpeck.ky = mKvecs.y;
       mNextSpeck.amp = mProbeAmp;
       mNextSpeck.phase = mPhaseList[mCurPhaseInd];
       mNextSpeck.isNull = false;

   }

}

void SpeckleBasic::nonProbeMeasurmentUpdate(double intensity, double variance){
    mProbeAmp = std::sqrt(intensity)*mDMCalFactor;

    if(mNProbeIters%10 == 0)
        BOOST_LOG_TRIVIAL(info) << "Iter: " << mNProbeIters << "; Probing at: " << mProbeAmp;

}

void SpeckleBasic::updateNullingSpeckle(){
    double ampFactor = mDMCalFactor*mDMCalFactor/mProbeAmp/4;
    double real = ampFactor*(mPhaseIntensities[0] - mPhaseIntensities[2])/mNProbeIters;
    double imag = ampFactor*(mPhaseIntensities[1] - mPhaseIntensities[3])/mNProbeIters;
    double avgVar = (mPhaseVars[0] + mPhaseVars[1] + mPhaseVars[2] + mPhaseVars[3])/2; 
    // 2 b/c re and im contain combos of measurements

    double nullingAmp = std::sqrt(real*real + imag*imag);
    double nullingSigma = std::sqrt(avgVar)*ampFactor/mNProbeIters; //this assumes sigma_re ~= sigma_im
    double nullingSNR = nullingAmp/nullingSigma;
    double nullingPhase = std::atan2(imag, real);
    nullingPhase += M_PI;
    nullingAmp *= mParams.get<double>("SpeckBasicParams.nullingGain");

    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": real: " << real;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": imag: " << imag;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": sigma: " << nullingSigma;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": SNR: " << nullingSNR;

    if(nullingSNR >= mParams.get<double>("SpeckBasicParams.snrThresh")){
        mNextSpeck.kx = mKvecs.x;
        mNextSpeck.ky = mKvecs.y;
        mNextSpeck.amp = nullingAmp;
        mNextSpeck.phase = nullingPhase;
        mNextSpeck.isNull = true;

        mNNullingIters++;

        BOOST_LOG_TRIVIAL(info) << "SpeckleBasic at " << mCoords << ": applying nulling speckle after " << mNProbeIters << " iterations";
        BOOST_LOG_TRIVIAL(info) << "                                   amplitude: " << nullingAmp;
        BOOST_LOG_TRIVIAL(info) << "                                   phase: " << nullingPhase;
        BOOST_LOG_TRIVIAL(info) << "                                   SNR: " << nullingSNR;

    }

    else{
        mNextSpeck.amp = 0;
        mNextSpeck.isNull = true;

    }

}

dmspeck SpeckleBasic::getNextSpeckle() const{ 
    return mNextSpeck;

}
