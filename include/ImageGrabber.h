#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <params.h>
#include <semaphore.h>
#include <stdlib.h>
#include <errno.h>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/ptree.hpp>
#include <mkidshm.h>

#ifndef IMAGEGRABBER_H
#define IMAGEGRABBER_H
/*
 * Interfaces with PacketMaster (MKID readout stream parsing code) to acquire images 
 * on-demand at a specified timestamp. Also does basic image processing
 * (dark/flat application and bad pixel masking). 
 */
class ImageGrabber
{
    private:
        cv::Mat mRawImageShm; //Stores raw MKID image. Wrapper for shared memory.
        cv::Mat mCtrlRegionImage; //Stores (processed) image of control region. 
        cv::Mat mBadPixMask; //Bad pixel mask (1 if bad)
        cv::Mat mBadPixMaskCtrl; //Bad pixel mask for control region
        cv::Mat mFlatWeights; //Image of flat weights (should multiply image by this)
        cv::Mat mFlatWeightsCtrl; //Image of flat weights in control region
        cv::Mat mDarkSub; //Image of dark counts (in cps)
        cv::Mat mDarkSubCtrl; //Image of dark counts for control region

        boost::property_tree::ptree mParams; //Object containing configuration options
        
        //sem_t *doneImgSem;
        //sem_t *takeImgSem;
        MKID_IMAGE mShmImage; //Shared memory buffer containing raw MKID image
        char *mBadPixBuff; //Buffer containing bad pixel mask
        char *mFlatCalBuff; //Buffer containing flat cal image
        char *mDarkSubBuff; //Buffer containing dark image
        int mXCenter; //x-coord of center PSF
        int mYCenter; //y-coord of center PSF
        int mXCtrlStart; 
        int mXCtrlEnd;
        int mYCtrlStart;
        int mYCtrlEnd; //Control region boundaries, in pixel coordinates relative to center PSF

        bool mUpToDate;

        double mIntegrationTime;

        void initialize();

        /**
        * Defines the boundaries of the control region, based on center location and coordinates specified in
        * mParams. Called by constructor and changeCenter()
        */
        void setCtrlRegion();

        /**
        * The following functions load in the specified calibration files
        **/
        void loadBadPixMask();
        void loadFlatCal();
        void loadDarkSub();

        /**
        * Applies bad pixel mask to control region. Currently a simple median filter on surrounding pixels.
        **/
        void badPixFiltCtrlRegion();

        /**
        * Applies gaussian blur w/ width lambda/D, normalized by blur of good pixel mask. Should have
        * better performance than simple bad pixel filter
        */
        //void gaussianBadPixFilt();
        
        /**
        * Applies flat calibration to full image. 
        * Element-wise multiplies mCtrlRegionImage by mFlatWeightsCtrl
        **/
        void applyFlatCal();

        /**
        * Applies flat calibration to control region. 
        * Element-wise multiplies mCtrlRegionImage by mFlatWeightsCtrl
        **/
        void applyFlatCalCtrlRegion();
        
        /**
        * Applies dark subtraction to full image.
        * Subtracts mDarkSubCtrl from mCtrlRegionImage.
        **/
        void applyDarkSub();

        /**
        * Applies dark subtraction to control region.
        * Subtracts mDarkSubCtrl from mCtrlRegionImage.
        **/
        void applyDarkSubCtrlRegion();

        /**
        * Runs image processing steps specified in config file on mCtrlRegionImage.
        **/
        void processCtrlRegion();

        /**
        * Runs image processing steps specified in config file on fullImage.
        **/
        void processFullImage();

        void close();

    public:
        /**
        * Constructor. Initializes (opens) shared memory spaces, semaphores, and cal arrays.
        * 
        * @param &ptree reference to a boost::property_tree object containing config options
        * @return ImageGrabber object
        */
        ImageGrabber(boost::property_tree::ptree &ptree);
        

        ImageGrabber(const ImageGrabber &other);
        ImageGrabber& operator=(const ImageGrabber &rhs);
        
        /**
        * Reads in the next image from shared memory (provided by PacketMaster in normal operation).
        * Waits for *doneImgSemPtr, then updates mRawImageShm with new image. Copies image into mCtrlRegionImage
        * and fullImage arrays.
        **/
        void readNextImage();


        /**
        * Sends signal to packetmaster (or simulation) to start taking an image;
        * i.e. increments *takeImgSemPtr. If interfacing w/ PacketMaster, image will consist of photons
        * tagged w/ timestamps between startts and startts + intTime, where intTime is specified in mParams.
        * @param startts timestamp in half-milliseconds since Jan 1 00:00 of current year. Placed in shmTs shared memory space.
        * @param integrationTime integration time in milliseconds
        */
        //void startIntegrating(uint64_t startts);
        void startIntegrating(uint64_t startts, double integrationTime);
        void startIntegrating(uint64_t startts, double integrationTime, int wvlStart, int wvlStop);

        /**
        * Returns the most recently taken image of the control region specified in mParams.
        * @return reference to image of control region (cv::Mat object w/ dtype CV_16UC1)
        */
        cv::Mat& getCtrlRegionImage(bool process=false);
        
        /**
        * Returns most recently taken image. 
        * NOTE: the returned object is just a wrapper for the shared memory space, so copy before modifying.
        * @return reference to array image (cv::Mat object w/ dtype CV_16UC)
        **/
        cv::Mat& getImage(bool process=false);


        const cv::Mat& getBadPixMask() const;
        const cv::Mat& getBadPixMaskCtrl() const;

        boost::property_tree::ptree getCfgParams() const;

        int getIntegrationTime() const;

        /**
        * Plots the image (or whatever you modify it to do!)
        * @param showControlRegion if true, plots just the control region, else plots the full image.
        **/
        void displayImage(bool makePlot=false);
        
        /**
        * Changes the location of the central PSF. This is important b/c control region is defined
        * wrt to PSF location.
        * @param xCent x coordinate of PSF
        * @param yCent y coordinate of PSF
        **/
        void changeCenter(int xCent, int yCent);



        ~ImageGrabber();

};
#endif
