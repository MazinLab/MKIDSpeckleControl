#include "SpeckleKalman.h"

SpeckleKalman::SpeckleKalman(cv::Point2d &pt, boost::property_tree::ptree &ptree):
        SpeckleController(pt, ptree){
    mProbeGridWidth = mParams.get<int>("KalmanParams.probeGridWidth");
    mNProbePos = mProbeGridWidth*mProbeGridWidth;

    mx = cv::Mat::zeros(2*mNProbePos, 1, CV_64F);
    mz = cv::Mat::zeros(2, 1, CV_64F);
    mK = cv::Mat::zeros(2*mNProbePos, 2, CV_64F);
    mH = cv::Mat::zeros(2, 2*mNProbePos, CV_64F);
    mA = cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mP = mParams.get<double>("KalmanParams.initStateVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mQ = mParams.get<double>("KalmanParams.processNoiseVar")*cv::Mat::eye(2*mNProbePos, 2*mNProbePos, CV_64F);
    mR = cv::Mat::zeros(2, 2, CV_64F);
    BOOST_LOG_TRIVIAL(info) << "Initialized KF matrices";

    correlateProcessNoise();
    BOOST_LOG_TRIVIAL(debug) << "Initial Process Noise: " << mP;

    initializeProbeGridKvecs();
    BOOST_LOG_TRIVIAL(debug) << "Probe Grid: " << mProbeGridKvecs;

}

void SpeckleKalman::correlateProcessNoise()
{
    cv::Mat realBlock = cv::Mat(mP, cv::Range(0, mNProbePos), cv::Range(0, mNProbePos));
    cv::Mat imagBlock = cv::Mat(mP, cv::Range(mNProbePos, 2*mNProbePos), cv::Range(mNProbePos, 2*mNProbePos));
    cv::Point2d ki, kj;
    int row, col;
    double kDist;

    for(int i = 0; i < mNProbePos; i++){
        for(int j = 0; j < mNProbePos; j++){
            std::tie(row, col) = getProbeGridIndices(i)
            ki = mProbeGridKvecs[row, col];
            std::tie(row, col) = getProbeGridIndices(i)
            kj = mProbeGridKvecs[row, col];
            kDist = cv::norm(ki - kj);
            
            realBlock[i, j] = std::sqrt(realBlock[i, i]*realBlock[j, j])*std::exp(-kDist*kDist/(2*mKvecCorrSigma*mKvecCorrSigma));

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
            mProbeGridKvecs[r, c] = kvec;

        }

    }

}
        
            

    

std::tuple<int, int> SpeckleKalman::getKalmanIndices(int r, int c)
{
    int realInd = r*mProbeGridWidth + c;
    int imagInd = realInd + mNProbePos;
    return std::make_tuple(realInd, imagInd);

}

std::tuple<int, int> SpeckleKalman::getProbeGridIndices(int kalmanInd)
{
    int row = kalmanInd/mProbeGridWidth;
    int col = kalmanInd%mProbeGridWidth;
    return std::make_tuple(row, col);

}

