#include "SpeckleController.h"

#ifndef SPECKLEKALMAN_H
#define SPECKLEKALMAN_H
class SpeckleKalman : public SpeckleController
{
    private:
        int mProbeGridWidth;
        int mNProbePos;
        double mKvecCorrSigma;
        double mProbeGridSpacing;
        cv::Mat_<cv::Point2d> mProbeGridKvecs; // k vector at each point 
        cv::Mat mProbeGridCounter; //dims: pgwxpgw - number of times each k has been probed (row,col) ind
        cv::Mat mProbeGridCorr; // correlation between pts at each "Kalman Index". dims: pgw^2*pgw^2
        cv::Point2d mCurProbePos;
        double mProbeAmp;
        dmspeck mNextSpeck;

        //Kalman filter matricies
        cv::Mat mP; // Process noise covariance
        cv::Mat mK; // Gain matrix
        cv::Mat mQ; // Process noise matrix
        cv::Mat mH; // Observation matrix
        cv::Mat mx; // State vector (real/imag part of speckle at each probe position)
        cv::Mat mA; // State transition matrix (should be I)

        cv::Mat mz; // Measurement vector (2D; real/imag phase intensities)
        cv::Mat mR; // Measurement noise covariance matrix

        // Makes process noise matrix correlated s.t. sig_i*sig_j = Cij*sig_i*sig_j
        void correlateProcessNoise();

        void initializeProbeGridKvecs();

        // Basically np.ravel_multi_index; gets the state vector indices (re and imag) of a given probe position
        std::tuple<int, int> getKalmanIndices(int r, int c);
        std::tuple<int, int> getProbeGridIndices(int kalmanInd);
        void updateKalmanState();

    public:
        SpeckleKalman(cv::Point2d pt, cv::Mat &image, boost::property_tree::ptree &ptree);

        void update(cv::Mat &image);

        dmspeck getNextSpeckle();

};
#endif
