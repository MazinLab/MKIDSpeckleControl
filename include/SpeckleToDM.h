#include <iostream>

#include "DMChannel.h"

#ifndef SPECKLETODM_H
#define SPECKLETODM_H
/*
 * Simple wrapper for speckles on the DM. 
 */
typedef struct
{
    double kx;
    double ky;
    double amp;
    double phase

} dmspeck;

/*
 * High level interface to DM - can add/remove speckles
 * and (eventually) adapt to changing cal parameters. 
 * TODO: consider adding (non-CACAO) simulation mode
 */
class SpeckleToDM
{
    private:
        std::vector<dmspeck> probeSpeckles;
        std::vector<dmspeck> nullingSpeckles;
        cv::Mat probeMap;
        cv::Mat nullMap;
        DMChannel dmChannel;

    public:
        
        /*
         * Constructor. Instantiates DMChannel for CACAO communication with 
         * dmChannel.
         * @param dmChan Name of dmChannel shared memory buffer to use
         * @param ptree boost property tree containing global config params
         */
        SpeckleToDM(const char *dmChan, boost::property_tree::ptree &ptree);

        /*
         * Add a probe speckle to DM. Only difference between probe and nulling
         * speckle is that probe will get cleared when clearProbeSpeckles() is 
         * called. DM map will only be changed when updateDM() is called
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

}
#endif
