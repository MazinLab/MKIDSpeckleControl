#include "SpeckleController.h"

#ifndef SPECKLEKALMAN_H
#define SPECKLEKALMAN_H
class SpeckleKalman : public SpeckleController
{
    private:
        double mPhaseList[NPHASES];
        double mPhaseIntensities[NPHASES];
        double mPhaseVars[NPHASES];
        double mLastIntTime; //TODO: figure out how to set this

        int mProbeGridWidth;
        int mNProbePos;
        int mMinProbeIters;
        double mKvecCorrSigma;
        double mProbeGridSpacing;
        cv::Mat_<cv::Point2d> mProbeGridKvecs; // k vector at each point 
        cv::Mat mProbeGridCounter; //dims: pgwxpgw - number of times each k has been probed (row,col) ind
        cv::Mat mProbeGridCorr; // correlation between pts at each "Kalman Index". dims: pgw^2*pgw^2
        cv::Point2i mCurProbePos; // position on kvec probe grid
        double mProbeAmp;
        double mMinProbeAmp;
        double mDMCalFactor;
        double mNullingGain;
        dmspeck mNextSpeck;
        cv::Ptr<cv::Formatter> mCVFormatter;

        //Kalman filter matricies
        cv::Mat mP; // Process noise covariance
        cv::Mat mK; // Gain matrix
        cv::Mat mQ; // Process noise matrix
        cv::Mat mQc; // "Effective" process noise from control/cal uncertainty
        cv::Mat mH; // Observation matrix
        cv::Mat mx; // State vector (real/imag part of speckle at each probe position)
        cv::Mat mA; // State transition matrix (should be I)

        cv::Mat mz; // Measurement vector (2D; real/imag phase intensities)
        cv::Mat mR; // Measurement noise covariance matrix

        // METHODS
        
        // required implementation of pure virtual methods
        void nonProbeMeasurementUpdate(double intensity, double variance);
        void probeMeasurementUpdate(int phaseInd, double intensity, double variance);
        dmspeck getNextProbeSpeckle(int phaseInd);
        dmspeck endOfProbeUpdate();

        // Makes process noise matrix correlated s.t. sig_i*sig_j = Cij*sig_i*sig_j
        void correlateProcessNoise(cv::Mat &noiseMat);

        void initializeProbeGridKvecs();

        

        // basically np.ravel_multi_index; gets the state vector indices (re and imag) of a given probe position
        std::tuple<int, int> getKalmanIndices(int r, int c);
        std::tuple<int, int> getKalmanIndices(cv::Point2i &probePos);
        std::tuple<int, int> getProbeGridIndices(int kalmanInd);


        void updateKalmanState();
        dmspeck updateNullingSpeckle();
        void updateProbeGridIndices();


    public:
        SpeckleKalman(cv::Point2d pt, boost::property_tree::ptree &ptree);

        void update(const cv::Mat &image, double integrationTime);

        dmspeck getNextSpeckle() const;

};
#endif
