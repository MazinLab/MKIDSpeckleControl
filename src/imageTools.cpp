#include "imageTools.h"

cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMask, int usFactor, double lambdaOverD)
{
    assert(image.rows == badPixMask.rows);
    assert(image.cols == badPixMask.cols);
    //remove bad pixels
    
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: remove bad pixels";
    cv::Mat badPixMaskInv = (~badPixMask)&1;
    badPixMaskInv.convertTo(badPixMaskInv, image.type());
    //image = image.mul(badPixMaskInv);
    cv::multiply(image, badPixMaskInv, image);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done remove bad pixels";

    //upsample
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: upsample";
    cv::Mat imageUS, badPixMaskInvUS;
    cv::resize(image, imageUS, cv::Size(0,0), usFactor, usFactor, cv::INTER_NEAREST);
    cv::resize(badPixMaskInv, badPixMaskInvUS, cv::Size(0,0), usFactor, usFactor, cv::INTER_NEAREST);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done upsample";

    //gaussian blur
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: gaussian blur";
    cv::GaussianBlur(imageUS, imageUS, cv::Size(7,7), lambdaOverD*(double)usFactor*0.42);
    cv::GaussianBlur(badPixMaskInvUS, badPixMaskInvUS, cv::Size(7,7), lambdaOverD*(double)usFactor*0.42);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done initial gaussian blur";
    badPixMaskInvUS.setTo(100000, badPixMaskInvUS<=0.05); //replace with really big number so we're not dividing by something small
    image = imageUS.mul(1/badPixMaskInvUS);
    BOOST_LOG_TRIVIAL(trace) << "bpfilt: done gaussian blur";
    return image;

}

