#include <opencv2/opencv.hpp>
#include <semaphore.h>
#include <mkidshm.h>
#include <ImageStreamIO.h>
#include <ImageStruct.h>
#include "MKIDImageSim.cpp"

//logging
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>



int main(){
    char dmShmImName[80] = "dm03disp09";
    char mkidShmName[80] = "DMToMKIDSim0";
    int fpNRows = 1400;
    int fpNCols = 1460;
    float nLDPerPix = 3;
    int nIntegrations = 1;
    int dmSemInd = 0;

    bool takingImage = false;
    int intCounter = 0;

    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

    //wrapper around shared memory data
    IMAGE dmShmIm;
    MKID_IMAGE fpShmIm;

    MKIDImageSim mkid(fpNRows, fpNCols, nLDPerPix);

    ImageStreamIO_openIm(&dmShmIm, dmShmImName);
    int dmNRows = dmShmIm.md->size[1];
    int dmNCols = dmShmIm.md->size[0];
    cv::Mat dmImMat(dmNRows, dmNCols, CV_32F, dmShmIm.array.F); 

    
    MKIDShmImage_open(&fpShmIm, mkidShmName);
    cv::Mat fpImMat(fpNRows, fpNCols, CV_32S, fpShmIm.image);
    
    while(true){
        if(sem_trywait(fpShmIm.takeImageSem)==0){
            BOOST_LOG_TRIVIAL(debug) << "Starting Image";
            fpImMat.setTo(0);
            takingImage = true;
            intCounter = 0;

        }

        if(takingImage){
            BOOST_LOG_TRIVIAL(debug) << "Waiting for DM";
            ImageStreamIO_semwait(&dmShmIm, dmSemInd);
            fpImMat += mkid.convertDMToFP(dmImMat);
            intCounter++;
            if(intCounter==nIntegrations){
                BOOST_LOG_TRIVIAL(debug) << "Done with image";
                MKIDShmImage_postDoneSem(&fpShmIm, -1);
                takingImage = false;

            }

        }

        else{
            ImageStreamIO_semtrywait(&dmShmIm, dmSemInd);

        }
            

    }

}
