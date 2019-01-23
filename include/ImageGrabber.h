#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <params.h>
#include <semaphore.h>
#include <stdlib.h>
#include <errno.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#define DONEIMGSEM "/speckNullDoneImg"
#define TAKEIMGSEM "/speckNullTakeImg"

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
        cv::Mat rawImageShm; //Stores raw MKID image. Wrapper for shared memory.
        cv::Mat ctrlRegionImage; //Stores (processed) image of control region. 
        cv::Mat fullImage; //Stores processed image. 
        cv::Mat badPixMask; //Bad pixel mask (1 if bad)
        cv::Mat badPixMaskCtrl; //Bad pixel mask for control region
        cv::Mat flatWeights; //Image of flat weights (should multiply image by this)
        cv::Mat flatWeightsCtrl; //Image of flat weights in control region
        cv::Mat darkSub; //Image of dark counts (in cps)
        cv::Mat darkSubCtrl; //Image of dark counts for control region
        boost::interprocess::shared_memory_object shmImgBuffer; //Shared memory object for raw image
        boost::interprocess::mapped_region imgBufferRegion; //Memory mapped region object for raw image
        boost::interprocess::shared_memory_object shmTs; //Shared memory object for start timestamp
        boost::interprocess::mapped_region tsMemRegion; //Memory mapped region object for start timestamp
        boost::interprocess::shared_memory_object shmIntTime; //Shared memory object for integration time
        boost::interprocess::mapped_region intTimeMemRegion; //Memory mapped region object for integration time
        boost::interprocess::named_semaphore *doneImgSemPtr; //Pointer to "done image" semaphore
        boost::interprocess::named_semaphore *takeImgSemPtr; //Pointer to "take image" semaphore

        boost::property_tree::ptree cfgParams; //Object containing configuration options
        
        //sem_t *doneImgSem;
        //sem_t *takeImgSem;
        uint16_t *imgArr; //Shared memory buffer containing raw MKID image
        uint64_t *tsPtr; //Shared memory buffer containing start timestamp for image
        uint64_t *intTimePtr; //Shared memory buffer containing integration time (in half ms)
        char *badPixArr; //Buffer containing bad pixel mask
        char *flatCalArr; //Buffer containing flat cal image
        char *darkSubArr; //Buffer containing dark image
        int xCenter; //x-coord of center PSF
        int yCenter; //y-coord of center PSF
        int xCtrlStart; 
        int xCtrlEnd;
        int yCtrlStart;
        int yCtrlEnd; //Control region boundaries, in pixel coordinates relative to center PSF

    public:
        /**
        * Constructor. Initializes (opens) shared memory spaces, semaphores, and cal arrays.
        * 
        * @param &ptree reference to a boost::property_tree object containing config options
        * @return ImageGrabber object
        */
        ImageGrabber(boost::property_tree::ptree &ptree);
        
        /**
        * Default constructor. Placeholder, do not use
        */
        ImageGrabber();
        
        /**
        * Reads in the next image from shared memory (provided by PacketMaster in normal operation).
        * Waits for *doneImgSemPtr, then updates rawImageShm with new image. Copies image into ctrlRegionImage
        * and fullImage arrays.
        **/
        void readNextImage();

        /**
        * Runs image processing steps specified in config file on ctrlRegionImage.
        **/
        void processCtrlRegion();

        /**
        * Runs image processing steps specified in config file on fullImage.
        **/
        void processFullImage();

        /**
        * Sends signal to packetmaster (or simulation) to start taking an image;
        * i.e. increments *takeImgSemPtr. If interfacing w/ PacketMaster, image will consist of photons
        * tagged w/ timestamps between startts and startts + intTime, where intTime is specified in cfgParams.
        * @param startts timestamp in seconds since Jan 1 00:00 of current year. Placed in shmTs shared memory space.
        */
        void startIntegrating(uint64_t startts);

        /**
        * Returns the most recently taken image of the control region specified in cfgParams.
        * @return reference to image of control region (cv::Mat object w/ dtype CV_16UC1)
        */
        cv::Mat& getCtrlRegionImage();
        
        /**
        * Returns most recently taken image. 
        * NOTE: the returned object is just a wrapper for the shared memory space, so copy before modifying.
        * @return reference to array image (cv::Mat object w/ dtype CV_16UC)
        **/
        cv::Mat& getRawImageShm();

        /**
        * Returns the most recently taken image
        **/
        cv::Mat& getFullImage();

        cv::Mat& getBadPixMask();
        cv::Mat& getBadPixMaskCtrl();

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

    private:
        /**
        * Assigns ctrlRegionImage to relevant subset of the raw image. Doesn't allocate any memory;
        * ctrlRegionImage becomes wrapper around a subset of the shared memory.
        **/
        void grabControlRegion();
        
        /**
        * Assigns ctrlRegionImage to relevant subset of the raw image. Creates a copy of the subarray,
        * unlike grabControlRegion().
        **/
        void copyControlRegionFromShm();

        /**
        * Assigns fullImage to the raw image in shared memory. Creates a copy of the subarray,
        * unlike grabControlRegion().
        **/
        void copyFullImageFromShm();
        
        /**
        * Defines the boundaries of the control region, based on center location and coordinates specified in
        * cfgParams. Called by constructor and changeCenter()
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
        * Element-wise multiplies ctrlRegionImage by flatWeightsCtrl
        **/
        void applyFlatCal();

        /**
        * Applies flat calibration to control region. 
        * Element-wise multiplies ctrlRegionImage by flatWeightsCtrl
        **/
        void applyFlatCalCtrlRegion();
        
        /**
        * Applies dark subtraction to full image.
        * Subtracts darkSubCtrl from ctrlRegionImage.
        **/
        void applyDarkSub();

        /**
        * Applies dark subtraction to control region.
        * Subtracts darkSubCtrl from ctrlRegionImage.
        **/
        void applyDarkSubCtrlRegion();

};
#endif
