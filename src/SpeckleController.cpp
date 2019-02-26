#include "SpeckleController.h"

SpeckleController::SpeckleController(cv::Point2d &pt, boost::property_tree::ptree &ptree)
{
    mParams = ptree;
    
    mCoords = pt;
    mKvecs = calculateKvecs(mCoords, mParams);

    for(int i=0; i<NPHASES; i++)
        phaseList[i] = (double)2*M_PI*i/NPHASES;

        
    apertureMask = cv::Mat::zeros(2*mParams.get<int>("NullingParams.apertureRadius")+1, 2*mParams.get<int>("NullingParams.apertureRadius")+1, CV_64F);
    cv::circle(apertureMask, cv::Point(mParams.get<int>("NullingParams.apertureRadius"), mParams.get<int>("NullingParams.apertureRadius")), mParams.get<int>("NullingParams.apertureRadius"), 1, -1);

    mIntensityCorrectionFactor = measureIntensityCorrection();


}

void updateBadPixMask(cv::Mat &mask)
{
    mBadPixMask = mask;
    mIntensityCorrectionFactor = measureIntensityCorrection();

}


std::tuple<double, double> SpeckleController::measureSpeckleIntensityAndSigma(cv::Mat &image)
{
    double measIntensity, measSigmaI;
    cv::Mat speckleIm = cv::Mat(image, cv::Range((int)mCoords.y-mParams.get<int>("NullingParams.apertureRadius"), 
        (int)mCoords.y+mParams.get<int>("NullingParams.apertureRadius")+1), cv::Range(mCoords.x-mParams.get<int>("NullingParams.apertureRadius"), 
        mCoords.x+mParams.get<int>("NullingParams.apertureRadius")+1));
    speckleIm = speckleIm.mul(apertureMask);
    //std::cout << speckleIm << std::endl;
    measIntensity = (double)cv::sum(speckleIm)[0]/mIntensityCorrectionFactor;
    measSigmaI = std::sqrt(measIntensity*mIntensityCorrectionFactor)/mIntensityCorrectionFactor;

    return std::make_tuple(measIntensity, measSigmaI);

}


double SpeckleController::measureIntensityCorrection()
{
    cv::Mat goodPixMask = (~mBadPixMask)&1;
    cv::Mat apertureGoodPixMask = cv::Mat(goodPixMask, cv::Range((int)mCoords.y-mParams.get<int>("NullingParams.apertureRadius"), 
        (int)mCoords.y+mParams.get<int>("NullingParams.apertureRadius")+1), cv::Range(mCoords.x-mParams.get<int>("NullingParams.apertureRadius"), 
        mCoords.x+mParams.get<int>("NullingParams.apertureRadius")+1));
    apertureGoodPixMask.convertTo(apertureGoodPixMask, CV_64F);
    cv::Mat gaussKernel = cv::getGaussianKernel(2*mParams.get<int>("NullingParams.apertureRadius")+1, mParams.get<double>("ImgParams.lambdaOverD")*0.42);
    gaussKernel = gaussKernel*gaussKernel.t();
    return (double)cv::sum(gaussKernel.mul(apertureGoodPixMask))[0]/cv::sum(gaussKernel)[0];

}


cv::Point2d SpeckleController::getCoordinates() {return mCoords;}

cv::Point2d SpeckleController::getKvecs() {return mKvecs;}

