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
    char dmShmImName[80] = "dm04disp00";
    char mkidShmName[80] = "DMCalTest0";
    int fpNRows;
    int fpNCols;
    float nLDPerPix = 3;
    int nIntegrations = 1;
    int dmSemInd = 5;

    bool takingImage = false;
    int intCounter = 0;

    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

    //wrapper around shared memory data
    IMAGE dmShmIm;
    MKID_IMAGE fpShmIm;


    ImageStreamIO_openIm(&dmShmIm, dmShmImName);
    int dmNRows = dmShmIm.md->size[1];
    int dmNCols = dmShmIm.md->size[0];
    cv::Mat dmImMat(dmNRows, dmNCols, CV_32F, dmShmIm.array.F); 

    
    MKIDShmImage_open(&fpShmIm, mkidShmName);
    fpNRows = fpShmIm.md->nRows;
    fpNCols = fpShmIm.md->nCols;
    cv::Mat fpImMat(fpNRows, fpNCols, CV_32S, fpShmIm.image);

    MKIDImageSim mkid(fpNRows, fpNCols, nLDPerPix);

    int semctr = 0;

    while(MKIDShmImage_checkIfDone(&fpShmIm, 0)==0){semctr++;}

    while(sem_trywait(fpShmIm.takeImageSem)==0){semctr++;}

    std::cout << "semctr: " << semctr << std::endl;
    
    while(true){
        if(sem_wait(fpShmIm.takeImageSem)==0){
            BOOST_LOG_TRIVIAL(debug) << "Starting Image";
            fpImMat.setTo(0);
            takingImage = true;
            intCounter = 0;
            //MKIDShmImage_setValid(&fpShmIm);
            fpShmIm.md->valid = 1;

        }

        if(takingImage){
            BOOST_LOG_TRIVIAL(debug) << "Grabbing data from DM...";
            fpImMat += mkid.convertDMToFP(dmImMat);
            intCounter++;
            if(intCounter==nIntegrations){
                BOOST_LOG_TRIVIAL(debug) << "Done with image";
                MKIDShmImage_postDoneSem(&fpShmIm, -1);
                takingImage = false;

            }

            else{
                ImageStreamIO_semwait(&dmShmIm, dmSemInd);
                BOOST_LOG_TRIVIAL(debug) << "Waiting for DM";

            }

        }

            

    }

}
