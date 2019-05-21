#include <opencv2/opencv.hpp>
#include <iostream>
#include <params.h>
#include <assert.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <boost/property_tree/ptree.hpp>
#include <boost/log/trivial.hpp>
//#include "simInterfaceTools.h"
#include "imageTools.h"
#include "dmTools.h"
#include "dmspeck.h"

#ifndef SPECKLECONTROLLER_H
#define SPECKLECONTROLLER_H

/**
 * Abstract class for implementing a speckle controller for 
 * a single speckle. Includes high level interface and some 
 * low level functionality. Subclasses should only interface 
 * using the provided function headings; i.e. they should not
 * define any additional public methods. 
 * Typical usage:
 *   - Instantiate
 *   - Get new DM probe/null /w getNextSpeckle()
 *   - Supply new measurement with update(&image)
 *   - Delete when speckle is sufficiently suppressed
 **/
class SpeckleController
{
    protected:
        boost::property_tree::ptree mParams; //container used to store configuration parameters

        double mPhaseList[NPHASES]; //List of probe phases
        double mPhaseIntensities[NPHASES];
        double mPhaseVars[NPHASES];
        double mInitialIntensity;
        double mInitialVar;

        int mNProbeIters;

        double mIntensityCorrectionFactor;
        cv::Mat mApertureMask; //Aperture window used to measure speckle intensity
        cv::Mat mBadPixMask;

        cv::Point2d mCoords;
        cv::Point2d mKvecs; //speckle k-vectors (spatial angular frequencies)
        
        int mCurPhaseInd;
        dmspeck mLastSpeckle;
        int mLastIntTime;

        //METHODS

        /**
        * Measures the speckle intensity in the provided image, then calculates
        * sigma and sets measSpeckIntensity and measSigma
        * @param image Image to use for intensity measurement
        **/
        std::tuple<double, double> measureSpeckleIntensityAndSigma(const cv::Mat &image, double integrationTime);

        double measureIntensityCorrection() const;

    public:
        /**
        * Constructor. Calculates kvecs from provided array coordinates, and PSF coordinates 
        * provided in config params. 
        * @param pt Speckle coordinates on the array
        * @param ptree Property tree of config parameters
        **/
        SpeckleController(cv::Point2d pt, boost::property_tree::ptree &ptree);

        virtual void update(const cv::Mat &image, double integrationTime) = 0;

        virtual dmspeck getNextSpeckle() const = 0;

        void updateBadPixMask(const cv::Mat &mask);

        /**
        * Return speckle location in pixel coordinates on the array.
        **/
        cv::Point2d getCoordinates() const;

        cv::Point2d getKvecs() const;

        int getNProbeIters() const;

        ~SpeckleController();


};
#endif
