#include "BadPixFilt.h"

BadPixFilt::BadPixFilt(int usFactor, double lambdaOverD, int kSize):
        mUSFactor(usFactor), mGaussKSize(kSize){
    mGaussSigma = KERNEL_SIGMA_FACTOR*(double)mUSFactor*lambdaOverD;

}

void BadPixFilt::updateBadPixMask(const cv::Mat badPixMask){
    mBadPixMask = badPixMask.clone();

    mBadPixMaskInv = (~mBadPixMask)&1;
    mBadPixMaskInv.convertTo(mBadPixMaskInv, CV_32F);

    cv::resize(mBadPixMaskInv, mBadPixMaskInvUS, cv::Size(0,0), mUSFactor, mUSFactor, cv::INTER_NEAREST);

    cv::GaussianBlur(mBadPixMaskInvUS, mBadPixMaskInvUSBlur, cv::Size(mGaussKSize, mGaussKSize), mGaussSigma);
    mBadPixMaskInvUSBlur.setTo(100000, mBadPixMaskInvUSBlur<=0.05); //replace with really big number so we're not dividing by something small

}

cv::Mat BadPixFilt::filter(const cv::Mat image){
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: remove bad pixels";
    cv::multiply(image, mBadPixMaskInv, image);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done remove bad pixels";

    //upsample
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: upsample";
    cv::Mat imageUS;
    cv::resize(image, imageUS, cv::Size(0,0), mUSFactor, mUSFactor, cv::INTER_NEAREST);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done upsample";

    //gaussian blur
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: gaussian blur";
    cv::GaussianBlur(imageUS, imageUS, cv::Size(mGaussKSize, mGaussKSize), mGaussSigma);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done initial gaussian blur";
    imageUS = imageUS.mul(1/mBadPixMaskInvUSBlur);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done gaussian blur";
    return imageUS;

}


        
//
//cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMaskInv, cv::Mat &badPixMaskInvUS, int usFactor, double lambdaOverD)
//{
//    assert(image.rows == badPixMaskInv.rows);
//    assert(image.cols == badPixMaskInv.cols);
//    //remove bad pixels
//    
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: remove bad pixels";
//    cv::multiply(image, badPixMaskInv, image);
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done remove bad pixels";
//
//    //upsample
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: upsample";
//    cv::Mat imageUS;
//    cv::resize(image, imageUS, cv::Size(0,0), usFactor, usFactor, cv::INTER_NEAREST);
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done upsample";
//
//    //gaussian blur
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: gaussian blur";
//    cv::GaussianBlur(imageUS, imageUS, cv::Size(7,7), lambdaOverD*(double)usFactor*0.42);
//    image = imageUS.mul(1/badPixMaskInvUS);
//    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done gaussian blur";
//    return image;
//
//}

