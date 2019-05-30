#include "SpeckleController.h"

SpeckleController::SpeckleController(cv::Point2d pt, boost::property_tree::ptree &ptree):
        mParams(ptree), mCoords(pt), mCurPhaseInd(-1){
    mKvecs = calculateKvecs(mCoords, mParams);

    BOOST_LOG_TRIVIAL(info) << "Creating new speckle at " << mCoords << ": kvecs: " << mKvecs;

    mApertureMask = cv::Mat::zeros(2*mParams.get<int>("NullingParams.apertureRadius")+1, 2*mParams.get<int>("NullingParams.apertureRadius")+1, CV_64F);
    cv::circle(mApertureMask, cv::Point(mParams.get<int>("NullingParams.apertureRadius"), mParams.get<int>("NullingParams.apertureRadius")), mParams.get<int>("NullingParams.apertureRadius"), 1, -1);

    mBadPixMask = cv::Mat::zeros(mParams.get<int>("ImgParams.yCtrlEnd") - mParams.get<int>("ImgParams.yCtrlStart"), 
        mParams.get<int>("ImgParams.xCtrlEnd") - mParams.get<int>("ImgParams.xCtrlStart"), CV_16U);
    mIntensityCorrectionFactor = 1;

    mNProbeIters = 0;
    mNNullingIters = 0;
    BOOST_LOG_TRIVIAL(debug) << "Speckle: done initialization";


}

void SpeckleController::updateBadPixMask(const cv::Mat &mask)
{
    mBadPixMask = mask;
    mIntensityCorrectionFactor = measureIntensityCorrection();
    BOOST_LOG_TRIVIAL(debug) << "Speckle: intensity correction factor: " << mIntensityCorrectionFactor;

}

void SpeckleController::update(const cv::Mat &image, double integrationTime){
    BOOST_LOG_TRIVIAL(debug) << "SpeckleBasic at " << mCoords << ": mCurPhaseInd: " << mCurPhaseInd;
    double intensity, variance;
    std::tie(intensity, variance) = measureSpeckleIntensityAndSigma(image, integrationTime);

    if(mCurPhaseInd == -1){
        nonProbeMeasurementUpdate(intensity, variance);

    }

    else{
        double intensity, variance;
        std::tie(intensity, variance) = measureSpeckleIntensityAndSigma(image, integrationTime);
        probeMeasurementUpdate(mCurPhaseInd, intensity, variance);

    }
         
   if(mCurPhaseInd == NPHASES-1){
       mNProbeIters++;
       mNextSpeck = endOfProbeUpdate();
       if(mNextSpeck.amp != 0)
           mNNullingIters++;
       mCurPhaseInd = -1;

   }

   else{
       mCurPhaseInd += 1;
       mNextSpeck = getNextProbeSpeckle(mCurPhaseInd);
       mNextSpeck.isNull = false;

   }

}


std::tuple<double, double> SpeckleController::measureSpeckleIntensityAndSigma(const cv::Mat &image, double integrationTime)
{
    double measIntensity, measVariance;
    cv::Mat speckleIm = cv::Mat(image, cv::Range((int)mCoords.y-mParams.get<int>("NullingParams.apertureRadius"), 
        (int)mCoords.y+mParams.get<int>("NullingParams.apertureRadius")+1), cv::Range(mCoords.x-mParams.get<int>("NullingParams.apertureRadius"), 
        mCoords.x+mParams.get<int>("NullingParams.apertureRadius")+1));
    speckleIm = speckleIm.mul(mApertureMask);
    measIntensity = (double)cv::sum(speckleIm)[0]/mIntensityCorrectionFactor*1000/integrationTime;
    measVariance = measIntensity*mIntensityCorrectionFactor/mIntensityCorrectionFactor*1000/integrationTime;

    BOOST_LOG_TRIVIAL(debug) << "Speckle at " << mCoords << ": intensity :" << measIntensity;
    BOOST_LOG_TRIVIAL(debug) << "Speckle at " << mCoords << ": variance:     " << measVariance;
    BOOST_LOG_TRIVIAL(trace) << "Speckle at " << mCoords << ": image:   \n" << speckleIm;
    BOOST_LOG_TRIVIAL(debug) << "";

    return std::make_tuple(measIntensity, measVariance);

}


double SpeckleController::measureIntensityCorrection() const
{
    cv::Mat goodPixMask = (~mBadPixMask)&1;
    cv::Mat apertureGoodPixMask = cv::Mat(goodPixMask, cv::Range((int)mCoords.y-mParams.get<int>("NullingParams.apertureRadius"), 
        (int)mCoords.y+mParams.get<int>("NullingParams.apertureRadius")+1), cv::Range((int)mCoords.x-mParams.get<int>("NullingParams.apertureRadius"), 
        (int)mCoords.x+mParams.get<int>("NullingParams.apertureRadius")+1));
    apertureGoodPixMask.convertTo(apertureGoodPixMask, CV_64F);
    cv::Mat gaussKernel = cv::getGaussianKernel(2*mParams.get<int>("NullingParams.apertureRadius")+1, mParams.get<double>("ImgParams.lambdaOverD")*0.42);
    gaussKernel = gaussKernel*gaussKernel.t();
    BOOST_LOG_TRIVIAL(trace) << "Speckle good pix mask: " <<  apertureGoodPixMask;
    return (double)cv::sum(gaussKernel.mul(apertureGoodPixMask))[0]/cv::sum(gaussKernel)[0];

}

dmspeck SpeckleController::getNextSpeckle() const {return mNextSpeck;}

cv::Point2d SpeckleController::getCoordinates() const {return mCoords;}

cv::Point2d SpeckleController::getKvecs() const {return mKvecs;}

int SpeckleController::getNProbeIters() const {return mNProbeIters;}

int SpeckleController::getNNullingIters() const {return mNNullingIters;}

SpeckleController::~SpeckleController(){
    //if(mNProbeIters > 0)
    //    BOOST_LOG_TRIVIAL(info) << "Deleting speckle at " << mCoords;

}
