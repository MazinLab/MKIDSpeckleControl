#include "SpeckleController.h"

#ifndef SPECKLEBASIC_H
#define SPECKLEBASIC_H

class SpeckleBasic : public SpeckleController {
    private:
        double mPhaseList[NPHASES];
        double mPhaseIntensities[NPHASES];
        double mPhaseVars[NPHASES];

        dmspeck mNextSpeck;
        double mProbeAmp;
        double mDMCalFactor;

        // Required declarations of virtual functions
        void nonProbeMeasurementUpdate(double intensity, double sigma);
        void probeMeasurementUpdate(int phaseInd, double intensity, double sigma);
        dmspeck getNextProbeSpeckle(int phaseInd);
        dmspeck endOfProbeUpdate();

    public:
        SpeckleBasic(cv::Point2d pt, boost::property_tree::ptree &cfgParams);

};
#endif
