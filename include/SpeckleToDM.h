#include <iostream>
#include <opencv2/opencv.hpp>

#include "DMChannel.h"
#include "dmspeck.h"

#ifndef SPECKLETODM_H
#define SPECKLETODM_H

#define DM_X_SIZE 50
#define DM_Y_SIZE 50

typedef float Pixel; //needed for DM map lambda function

/*
 * High level interface to DM - can add/remove speckles
 * and (eventually) adapt to changing cal parameters. 
 * TODO: consider adding (non-CACAO) simulation mode
 */
class SpeckleToDM
{
    private:
        std::vector<dmspeck> probeSpeckles; //not used
        std::vector<dmspeck> nullingSpeckles; //not used
        //cv::Mat probeMap;
        //cv::Mat nullMap;
        float *fullMapShm; //wrapper around shm image of DM channel
        float probeMap[DM_X_SIZE*DM_Y_SIZE];
        float nullMap[DM_X_SIZE*DM_Y_SIZE];
        float tempMap[DM_X_SIZE*DM_Y_SIZE];
        
        //cv::Mat tempMap;

        DMChannel dmChannel;
        //int dmXSize;
        //int dmYSize;

        bool usenm;

        boost::property_tree::ptree cfgParams;

        //Methods:
        void generateMapFromSpeckle(const cv::Point2d kvecs, double amp, double phase, float *map);

    public:
        
        /*
         * Constructor. Instantiates DMChannel for CACAO communication with 
         * DM channel.
         * @param dmChan Name of dmChannel shared memory buffer to use
         */
        //SpeckleToDM(const char dmChanName[80], boost::property_tree::ptree &ptree);
        SpeckleToDM(const char dmChanName[80], bool _usenm=true);

        /*
         * Add a probe speckle to DM. Only difference between probe and nulling
         * speckle is that probe will get cleared when clearProbeSpeckles() is 
         * called. DM map will only be changed when updateDM() is called
         * @param kvecs cv::Point containing kx and ky 
         * @param amp DM amplitude
         * @param phase DM phase
         */
        void addProbeSpeckle(cv::Point2d kvecs, double amp, double phase);

        /* Overloads above function. 
         * @param kx k-vector x
         * @param ky k-vector y
         * @param amp DM amplitude
         * @param phase DM phase
         */
        void addProbeSpeckle(double kx, double ky, double amp, double phase);

        /*
         * Add a nulling speckle to DM. Only difference between probe and nulling
         * speckle is that probe will get cleared when clearProbeSpeckles() is 
         * called. DM map will only be changed when updateDM() is called.
         * @param kvecs cv::Point containing kx and ky 
         * @param amp DM amplitude
         * @param phase DM phase
         */
        void addNullingSpeckle(cv::Point2d kvecs, double amp, double phase);

        /* Overloads above function. 
         * @param kx k-vector x
         * @param ky k-vector y
         * @param amp DM amplitude
         * @param phase DM phase
         */
        void addNullingSpeckle(double kx, double ky, double amp, double phase);

        /*
         * Clears probe speckles. DM map will only change when 
         * updateDM() is called.
         */
        void clearProbeSpeckles();

        /*
         * Clears probe speckles. DM map will only change when 
         * updateDM() is called.
         */
        void clearNullingSpeckles();

        /*
         * Updates the DM with all changes to the map since last update. (i.e.
         * DM map will be consistent with the lists probeSpeckles and nullingSpeckles)
         */
        void updateDM();

        /*
         * Updates the calibration parameters (probably just amplitude scaling) 
         * used to generate the DM map. Don't really know what this entails so
         * not yet implemented. Intended to deal w/ PyWFS calibration offset.
         */
        void updateCalParams(int placeholder);

        void setMapToZero(float *map);

        // Getters:
        int getXSize();
        int getYSize();

};
#endif
