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



int main(int argc, char* argv[]){
    char dmShmImName[80] = "dm04disp";
    char mkidShmName[80] = "DMCalTest0";
    char badPixMask[200] = "/home/neelay/data/20190514/finalMap_20181218_badPixMask.bin";
    int fpNRows;
    int fpNCols;
    float nLDPerPix = 3;
    int nIntegrations = 1;
    int dmSemInd = 5;

    bool takingImage = false;
    int intCounter = 0;

    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);

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
    if(argc==2)
        mkid = MKIDImageSim(fpNRows, fpNCols, nLDPerPix, badPixMask);

    int semctr = 0;

    while(MKIDShmImage_checkIfDone(&fpShmIm, 0)==0){semctr++;}
    while(sem_trywait(fpShmIm.takeImageSem)==0){semctr++;}
    while(ImageStreamIO_semtrywait(&dmShmIm, dmSemInd)==0){semctr++;};

    std::cout << "semctr: " << semctr << std::endl;
    semctr = 0;
    
    while(true){
        if(sem_wait(fpShmIm.takeImageSem)==0){
            while(ImageStreamIO_semtrywait(&dmShmIm, dmSemInd)==0){semctr++;};
            while(sem_trywait(fpShmIm.takeImageSem)==0){semctr++;}
            BOOST_LOG_TRIVIAL(debug) << "Starting Image";
            fpImMat.setTo(0);
            takingImage = true;
            intCounter = 0;
            //MKIDShmImage_setValid(&fpShmIm);
            fpShmIm.md->valid = 1;

        }

        if(takingImage){
            BOOST_LOG_TRIVIAL(debug) << "Grabbing data from DM...";
            fpImMat += mkid.convertDMToFP(dmImMat, fpShmIm.md->integrationTime/(2*nIntegrations));
            intCounter++;
            if(intCounter==nIntegrations){
                BOOST_LOG_TRIVIAL(debug) << "Done with image";
                MKIDShmImage_postDoneSem(&fpShmIm, -1);
                takingImage = false;

            }


        }

        BOOST_LOG_TRIVIAL(debug) << "Waiting for DM...";
        ImageStreamIO_semwait(&dmShmIm, dmSemInd);
        BOOST_LOG_TRIVIAL(debug) << "DM ACK, waiting for image to start";
            

    }

}
