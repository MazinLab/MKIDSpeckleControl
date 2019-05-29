#include "SpeckleController.h"

class SpeckleBasic : SpeckleController {
    public SpeckleBasic(cv::Point2d pt, boost::property_tree::ptree &cfgParams);

    private:
        dmspeck mNextSpeck;
        double mProbeAmp;
        double mDMCalFactor;

        void nonProbeMeasurmentUpdate(double intensity, double variance);
        void updateNullingSpeckle();

    public:
        void update(const cv::Mat &image, double integrationTime);

};
