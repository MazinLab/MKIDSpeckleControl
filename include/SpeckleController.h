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
#include "dmTools.h"
#include "dmspeck.h"

#ifndef SPECKLECONTROLLER_H
#define SPECKLECONTROLLER_H

/**
 * Abstract class for implementing a speckle controller for 
 * a single speckle. Defines public interface with 
 * the following usage procedure:
 *   - Instantiate
 *   - update with initial image using update(&image, integrationTime)
 *   - perform NPHASES probe measurements 
 *      - getNextSpeckle() followed by update(&image, integrationTime)
 *   - apply control output and update with result (no probes) and repeat
 *
 * Subclasses should only implement the specified virtual methods
 * (and any private methods/attributes required for this); no additional
 * public methods should be defined.
 *
 **/
class SpeckleController
{
    private:
        int mNProbeIters;
        int mNNullingIters;
        int mCurPhaseInd;
        cv::Mat mApertureMask; //Aperture window used to measure speckle intensity
        cv::Mat mBadPixMask;
        cv::Mat mSpeckleIm;
        dmspeck mNextSpeck;
        double mIntensityCorrectionFactor;

        double measureIntensityCorrection() const;

        /**
        * Measures the speckle intensity in the provided image, then calculates
        * sigma and sets measSpeckIntensity and measSigma
        * @param image Image to use for intensity measurement
        **/
        std::tuple<double, double> measureSpeckleIntensityAndSigma(const cv::Mat &image, double integrationTime);
        
        /**
         * Pure virtual. Called on the initial update and after control outputs (including 
         * nullingAmp=0).
         * @param intensity Intensity of speckle in last image
         * @param variance Variance of speckle intensity
         */
        virtual void nonProbeMeasurementUpdate(double intensity, double variance) = 0;

        /**
         * Pure virtual. Called after DM probe outputs.
         * @param phaseInd Index of probe used [0, NPHASES)
         * @param intensity Intensity of speckle in last image
         * @param variance Variance of speckle intensity
         */
        virtual void probeMeasurementUpdate(int phaseInd, double intensity, double variance) = 0;

        /**
         * Pure virtual. Called after final probe; returns control (nulling) speckle.
         */
        virtual dmspeck endOfProbeUpdate() = 0;

        /**
         * Pure virtual. Returns dmspeck specifying probe speckle at index phaseInd.
         */
        virtual dmspeck getNextProbeSpeckle(int phaseInd) = 0;

        

    protected:
        boost::property_tree::ptree mParams; //container used to store configuration parameters

        cv::Point2d mCoords; //speckle coordinates in control region
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
         * subclasses; i.e. control outputs (speck.isNull = true) should only be applied when
         * nIters%(NPHASES+1) = 0
         *
         * @param image Image of control region
         * @param integrationTime IntegrationTime in ms
         */
        void update(const cv::Mat &image, double integrationTime);

        /**
         * Returns a dmspeck containing the next (probe or null) speckle to apply to the 
         * DM. Speckle is completely determined in update(); this function is separate 
         * for flexibility
         */
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
