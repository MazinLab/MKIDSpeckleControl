#include <opencv2/opencv.hpp>
#include <random>

//logging
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

class MKIDImageSim{
    int fpRows;
    int fpCols;
    float nPixPerLD;
    float wvl; //microns
    cv::Mat badPixMask;
    cv::Mat flatCal;
    char *badPixBuff;
    bool useBadPixMask;

    public:

        MKIDImageSim(int r, int c, float ldPix, float wl=1){
            fpRows = r;
            fpCols = c;

            nPixPerLD = ldPix;
            wvl = wl;
            badPixBuff = NULL;
            useBadPixMask = false;

        }

        MKIDImageSim(int r, int c, float ldPix, const std::string badPixFile, float wl=1){
            fpRows = r;
            fpCols = c;

            nPixPerLD = ldPix;
            wvl = wl;

            badPixBuff = new char[sizeof(uint16_t)*fpRows*fpCols];
            loadBadPixMask(badPixFile);
            useBadPixMask = true;

        }
        
        // dmImShm: cv::Mat wrapper around DM shared memory buffer
        // integrationTime: in ms
        cv::Mat convertDMToFP(const cv::Mat &dmImShm, int integrationTime, bool addNoise=true, int cpsBackground=0){
            int ppRows = (int)dmImShm.rows*nPixPerLD;
            int ppCols = (int)dmImShm.cols*nPixPerLD;
            if((ppRows < fpRows) || (ppCols < fpCols)){
                BOOST_LOG_TRIVIAL(error) << "FP larger than PP not implemented";
                throw;

            }
            cv::Mat ppE(ppRows, ppCols, CV_32FC2);
            cv::Mat fpE(ppRows, ppCols, CV_32FC2); //complex float
            cv::Mat fpIm(ppRows, ppCols, CV_32F); //intensities
            cv::Mat dmE(dmImShm.rows, dmImShm.cols, CV_32FC2); 

            //cv::Mat dmIm((int)dmImShm.rows*nPixPerLD, (int)dmImShm.cols*nPixPerLD, CV_32F);
            //dmIm.setTo(0);
            //cv::Mat dmTmp = dmIm(cv::Range(dmIm.rows/2-dmImShm.rows/2, dmIm.rows/2+dmImShm.rows/2), 
            //            cv::Range(dmIm.cols/2-dmImShm.cols/2, dmIm.cols/2+dmImShm.cols/2));
            //dmImShm.copyTo(dmTmp);
            
            dmE.forEach<cv::Vec2f>([this, &dmImShm](cv::Vec2f &value, const int *position) -> void { 
                    value = cv::Vec2f(std::cos(2*M_PI*dmImShm.at<float>(position[0], position[1])/wvl) - 1, // -1 simulates coronagraph
                            std::sin(2*M_PI*dmImShm.at<float>(position[0], position[1])/wvl));

                }); //convert phase to complex E-field

            cv::Mat circAperture(dmE.rows, dmE.cols, CV_32FC2, cv::Scalar(0, 0));
            cv::circle(circAperture, cv::Point2d((double)dmE.cols/2, (double)dmE.rows/2), dmE.rows/2, cv::Scalar(1, 1), -1);
            dmE = dmE.mul(circAperture);

            ppE.setTo(0);
            cv::Mat ppeTmp = ppE.rowRange(ppE.rows/2-dmE.rows/2, 
                    ppE.rows/2+dmE.rows/2).colRange(ppE.cols/2-dmE.cols/2, ppE.cols/2+dmE.cols/2);
            dmE.copyTo(ppeTmp);
            
            cv::dft(ppE, fpE, cv::DFT_COMPLEX_OUTPUT); //fft
            
            //fix quadrants so 0 freq is at center
            int qx = ppE.cols/2;
            int qy = ppE.rows/2;
            cv::Mat q0(fpE, cv::Rect(0, 0, qx, qy));
            cv::Mat q1(fpE, cv::Rect(qx, 0, qx, qy));
            cv::Mat q2(fpE, cv::Rect(0, qy, qx, qy));
            cv::Mat q3(fpE, cv::Rect(qx, qy, qx, qy));
            cv::Mat tmp;
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);
            
            cv::Mat eField[2];
            cv::split(fpE, eField);
            cv::magnitude(eField[0], eField[1], fpIm);
            fpIm = fpIm.mul(fpIm);

            fpIm = fpIm/10; //normalization to make CPS values more appetizing
            fpIm = fpIm*integrationTime/1000; //int time in ms, convert fpIm from CPS to counts
            if(addNoise)
                addPoissonNoise(fpIm);
            if(cpsBackground > 0)
                addPoissonBackground(fpIm, (double)cpsBackground*integrationTime/1000); 
            cv::Mat fpImOut(ppRows, ppCols, CV_32S);
            fpIm.convertTo(fpImOut, CV_32S);
            fpImOut = cv::Mat(fpImOut, cv::Range(ppRows/2 - fpRows/2, ppRows/2 + fpRows/2), 
                    cv::Range(ppCols/2 - fpCols/2, ppCols/2 + fpCols/2));
            //std::cout << fpImOut.rows << " " << fpImOut.cols
            if(useBadPixMask)
                applyBadPixMask(fpImOut);

            BOOST_LOG_TRIVIAL(debug) << "integration time: " << integrationTime;
            return fpImOut;
            
        }

        void addPoissonNoise(cv::Mat &fpIm){
            std::default_random_engine generator; 

            fpIm.forEach<float>([this, &generator](float &value, const int *position) -> void {
                    std::poisson_distribution<int> dist(value);
                    value = (float)dist(generator);

                });

        }

        void addPoissonBackground(cv::Mat &fpIm, double counts){
            std::default_random_engine generator; 

            fpIm.forEach<float>([this, counts, &generator](float &value, const int *position) -> void {
                    std::poisson_distribution<int> dist(counts);
                    value += (float)dist(generator);

                });

        }

        void applyBadPixMask(cv::Mat &fpIm){
            fpIm = fpIm.mul((~badPixMask)&1);

        }

        void loadBadPixMask(const std::string badPixFn){ 
            std::ifstream badPixFile(badPixFn.c_str(), std::ifstream::in|std::ifstream::binary);
            if(!badPixFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find bad pixel mask";
            badPixFile.read(badPixBuff, sizeof(uint16_t)*fpRows*fpCols);
            badPixMask = cv::Mat(fpRows, fpCols, CV_16UC1, badPixBuff);
            badPixMask.convertTo(badPixMask, CV_32S);
            BOOST_LOG_TRIVIAL(trace) << "bad pixel mask \n" << badPixMask;

        }

        ~MKIDImageSim(){
            if(badPixBuff != NULL)
                delete badPixBuff;

        }

};
