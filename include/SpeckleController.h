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
    private:
        int mNProbeIters;
        int mNNullingIters;
        int mCurPhaseInd;
        cv::Mat mApertureMask; //Aperture window used to measure speckle intensity
        cv::Mat mBadPixMask;
        dmspeck mNextSpeck;
        double mIntensityCorrectionFactor;

        double measureIntensityCorrection() const;

        /**
        * Measures the speckle intensity in the provided image, then calculates
        * sigma and sets measSpeckIntensity and measSigma
        * @param image Image to use for intensity measurement
        **/
        std::tuple<double, double> measureSpeckleIntensityAndSigma(const cv::Mat &image, double integrationTime);

        virtual void nonProbeMeasurementUpdate(double intensity, double variance) = 0;

        virtual void probeMeasurementUpdate(int phaseInd, double intensity, double variance) = 0;

        // return nulling speckle
        virtual dmspeck endOfProbeUpdate() = 0;

        virtual dmspeck getNextProbeSpeckle(int phaseInd) = 0;

        

    protected:
        boost::property_tree::ptree mParams; //container used to store configuration parameters

        cv::Point2d mCoords;
        cv::Point2d mKvecs; //speckle k-vectors (spatial angular frequencies)
        


    public:
        /**
        * Constructor. Calculates kvecs from provided array coordinates, and PSF coordinates 
        * provided in config params. 
        * @param pt Speckle coordinates on the array
        * @param ptree Property tree of config parameters
        **/
        SpeckleController(cv::Point2d pt, boost::property_tree::ptree &ptree);

        /**
         * Update controller state with new image. It is assumed that the previous control
         * output was applied before taking the image. 
         *
         * A structure of NPHASES probe iters + 1 control output iter is imposed on all
         * subclasses; i.e. control outputs (speck.isNull = true) can only be applied when
         * nIters%(NPHASES+1) = 0
         *
         * @param image Image of control region
         * @param integrationTime IntegrationTime in ms
         */
        void update(const cv::Mat &image, double integrationTime);

        dmspeck getNextSpeckle() const;

        void updateBadPixMask(const cv::Mat &mask);

        /**
        * Return speckle location in pixel coordinates on the array.
        **/
        cv::Point2d getCoordinates() const;

        cv::Point2d getKvecs() const;

        int getNProbeIters() const;
        int getNNullingIters() const;

        ~SpeckleController();


};
#endif
