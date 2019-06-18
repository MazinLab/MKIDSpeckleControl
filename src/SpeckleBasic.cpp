#include "SpeckleBasic.h"

SpeckleBasic::SpeckleBasic(cv::Point2d pt, boost::property_tree::ptree &cfgParams): SpeckleController(pt, cfgParams){
    for(int i=0; i<NPHASES; i++){
        mPhaseIntensities[i] = 0;
        mPhaseVars[i] = 0;
        mPhaseList[i] = (double)2*M_PI*i/NPHASES;

    }

    mDMCalFactor = getDMCalFactorCPS(mKvecs, mParams.get<double>("DMParams.a"), mParams.get<double>("DMParams.b"), 
            mParams.get<double>("DMParams.c"));

    BOOST_LOG_TRIVIAL(debug) << "DM Cal Factor: " << mDMCalFactor;

}
        


void SpeckleBasic::nonProbeMeasurementUpdate(double intensity, double variance){
    mProbeAmp = mProbeAmp*getNProbeIters()/(getNProbeIters() + 1) 
        + std::sqrt(intensity)*mDMCalFactor/(getNProbeIters() + 1);

    if(getNProbeIters()%10 == 0)
        BOOST_LOG_TRIVIAL(info) << "Iter: " << getNProbeIters() << "; Probing at: " << mProbeAmp;

}

void SpeckleBasic::probeMeasurementUpdate(int phaseInd, double intensity, double variance){
   mPhaseIntensities[phaseInd] += intensity;
   mPhaseVars[phaseInd] += variance;

}

dmspeck SpeckleBasic::getNextProbeSpeckle(int phaseInd){
   dmspeck nextSpeck;
   nextSpeck.kx = mKvecs.x;
   nextSpeck.ky = mKvecs.y;
   nextSpeck.amp = mProbeAmp;
   nextSpeck.phase = mPhaseList[phaseInd];
   nextSpeck.isNull = false;
   return nextSpeck;

}

dmspeck SpeckleBasic::endOfProbeUpdate(){
    double ampFactor = mDMCalFactor*mDMCalFactor/mProbeAmp/4;
    double real = ampFactor*(mPhaseIntensities[0] - mPhaseIntensities[2])/getNProbeIters();
    double imag = ampFactor*(mPhaseIntensities[1] - mPhaseIntensities[3])/getNProbeIters();
    double avgVar = (mPhaseVars[0] + mPhaseVars[1] + mPhaseVars[2] + mPhaseVars[3])/2; 
    // 2 b/c re and im contain combos of measurements

    double nullingAmp = std::sqrt(real*real + imag*imag);
    double nullingSigma = std::sqrt(avgVar)*ampFactor/getNProbeIters(); //this assumes sigma_re ~= sigma_im
    double nullingSNR = nullingAmp/nullingSigma;
    double nullingPhase = std::atan2(imag, real);
    nullingPhase += M_PI;
    nullingAmp *= mParams.get<double>("SpeckBasicParams.nullingGain");

    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseIntensities: " << mPhaseIntensities[0];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseIntensities: " << mPhaseIntensities[1];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseIntensities: " << mPhaseIntensities[2];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseIntensities: " << mPhaseIntensities[3];

    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseVars: " << mPhaseVars[0];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseVars: " << mPhaseVars[1];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseVars: " << mPhaseVars[2];
    BOOST_LOG_TRIVIAL(trace) << "SpeckleBasic at " << mCoords << ": mPhaseVars: " << mPhaseVars[3];

    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": real: " << real;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": imag: " << imag;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": sigma: " << nullingSigma;
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": SNR: " << nullingSNR;

    dmspeck nullingSpeck;

    if(nullingSNR >= mParams.get<double>("SpeckBasicParams.snrThresh")){
        nullingSpeck.kx = mKvecs.x;
        nullingSpeck.ky = mKvecs.y;
        nullingSpeck.amp = nullingAmp;
        nullingSpeck.phase = nullingPhase;
        nullingSpeck.isNull = true;

        BOOST_LOG_TRIVIAL(info) << "SpeckleBasic at " << mCoords << ": applying nulling speckle after " << getNProbeIters() << " iterations";
        BOOST_LOG_TRIVIAL(info) << "                                   amplitude: " << nullingAmp;
        BOOST_LOG_TRIVIAL(info) << "                                   phase: " << nullingPhase;
        BOOST_LOG_TRIVIAL(info) << "                                   SNR: " << nullingSNR;

    }

    else{
        nullingSpeck.amp = 0;
        nullingSpeck.kx = mKvecs.x;
        nullingSpeck.ky = mKvecs.y;
        nullingSpeck.phase = nullingPhase;
        nullingSpeck.isNull = true;

    }

    return nullingSpeck;

}

