#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/log/trivial.hpp>

#define KERNEL_SIGMA_FACTOR 0.42

#ifndef _BADPIXFILT_H
#define _BADPIXFILT_H

class BadPixFilt
{
    private:
        cv::Mat mBadPixMask;
        cv::Mat mBadPixMaskInv;
        cv::Mat mBadPixMaskInvUS;
        cv::Mat mBadPixMaskInvUSBlur;

        int mUSFactor;
        int mGaussSigma;
        int mGaussKSize;

    public:
        BadPixFilt(int usFactor, double lambdaOverD, int kSize);
        void updateBadPixMask(const cv::Mat badPixMask);
        cv::Mat filter(const cv::Mat image);

};

#endif

