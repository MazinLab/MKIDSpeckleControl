#include "SpeckleController.h"

#ifndef SPECKLEBASIC_H
#define SPECKLEBASIC_H

class SpeckleBasic : public SpeckleController {
    private:
        dmspeck mNextSpeck;
        double mProbeAmp;
        double mDMCalFactor;

        void nonProbeMeasurmentUpdate(double intensity, double variance);
        void updateNullingSpeckle();

    public:
        SpeckleBasic(cv::Point2d pt, boost::property_tree::ptree &cfgParams);
        void update(const cv::Mat &image, double integrationTime);
        dmspeck getNextSpeckle() const;

};
#endif
