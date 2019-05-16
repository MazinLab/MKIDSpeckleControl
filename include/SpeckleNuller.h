#include <opencv2/opencv.hpp>
#include <iostream>
#include "params.h"
#include "ImageGrabber.h"
#include "SpeckleController.h"
#include "SpeckleKalman.h"
#include "SpeckleToDM.h"
#include "imageTools.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#define DELETE_SPECKLE -1

#ifndef SPECKLENULLER_H
#define SPECKLENULLER_H
/**
* Stores speckle coordinates and intensity. Mostly just for easy sorting in
* detectSpeckles method.
**/
struct ImgPt
{
    cv::Point2d coordinates;
    double intensity;

    // for sorting purposes only
    bool operator<(ImgPt &rPt){return intensity > rPt.intensity;}

};

typedef SpeckleKalman SpeckleCtrlClass;


/**
* Implements the speckle nulling loop. Has methods for detecting, probing, and
* nulling speckles. Contains instances of ImageGrabber and P3KCom, as well as 
* a list of Speckle objects.
**/
class SpeckleNuller
{
    private: 
        cv::Mat mImage; //Last MKID image obtained from PacketMaster (control region)
        cv::Mat mBadPixMask; //Bad pixel mask of image. Should be updated when ctrl region changes
        std::vector<SpeckleCtrlClass> mSpecklesList, mNullSpecklesList; //List of Speckle (objects) being nulled
        std::vector<dmspeck> mNextDMSpecks; //List of speckles to put on DM
        boost::property_tree::ptree mParams; //Container for configuration params
        SpeckleToDM mDM;
        int mIters;
        double mIntegrationTime;

        /**
         * Detects and centroids speckles in current image.
         * @return vector of ImgPt objects containing speckle coordinates and intensities
         **/
        std::vector<ImgPt> detectSpeckles();

        /**
         * Cuts out elements of maxImgPts that are within an exclusion zone 
         * of existing speckles or each-other.
         **/
        void exclusionZoneCut(std::vector<ImgPt> &maxImgPts, bool checkCurrentSpeckles);

        /**
         * Checks whether any nulled speckles have been re-detected. If so,
         * updates speckle state and puts back in speckles list. Removes
         * point from maxImgPts.
         **/
        void updateAndCutNulledSpeckles(std::vector<ImgPt> &maxImgPts);

        /**
         * Checks whether any un-nulled speckles have been re-detected
         * (i.e. are in the list maxImgPts).
         */
        void updateAndCutActiveSpeckles(std::vector<ImgPt> &maxImgPts);

        /**
         * Creates speckle objects from a list of ImgPts; stores these
         * in specklesList.
         */
        void createSpeckleObjects(std::vector<ImgPt> &imgPts, bool update=true);

        void findNewSpeckles();

        void updateSpeckles();

    public:
        /**
        * Constructor. Initializes ImageGrabber and P3KCom objects.
        **/
        SpeckleNuller(boost::property_tree::ptree &ptree);
 
        /**
        * Updates the current image (of ctrl region)
        * @param image New image of ctrl region
        **/
        void update(const cv::Mat &newImage, double integrationTime);

        void updateBadPixMask(const cv::Mat &newMask);

        void updateDM();

        void clearSpeckleObjects();


};
#endif
