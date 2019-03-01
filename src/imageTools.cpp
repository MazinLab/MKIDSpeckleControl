#include "imageTools.h"

cv::Mat gaussianBadPixUSFilt(cv::Mat image, cv::Mat &badPixMask, int usFactor, double lambdaOverD)
{
    //remove bad pixels
    cv::Mat badPixMaskInv = (~badPixMask)&1;
    badPixMaskInv.convertTo(badPixMaskInv, CV_64FC1);
    image = image.mul(badPixMaskInv);

    //upsample
    cv::Mat imageUS, badPixMaskInvUS;
    cv::resize(image, imageUS, cv::Size(0,0), usFactor, usFactor, cv::INTER_NEAREST);
    cv::resize(badPixMaskInv, badPixMaskInvUS, cv::Size(0,0), usFactor, usFactor, cv::INTER_NEAREST);

    //gaussian blur
    cv::GaussianBlur(imageUS, imageUS, cv::Size(0,0), lambdaOverD*(double)usFactor*0.42);
    cv::GaussianBlur(badPixMaskInvUS, badPixMaskInvUS, cv::Size(0,0), lambdaOverD*(double)usFactor*0.42);
    badPixMaskInvUS.setTo(100000, badPixMaskInvUS<=0.05); //replace with really big number so we're not dividing by something small
    image = imageUS.mul(1/badPixMaskInvUS);
    return image;

}

