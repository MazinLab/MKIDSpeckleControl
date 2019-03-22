#include <opencv2/opencv.hpp>

class MKIDImageSim{
    int nRows;
    int nCols;
    float nLDPerPix;
    float wvl; //microns
    cv::Mat badPixMask;
    cv::Mat flatCal;

    public:

        MKIDImageSim(int r, int c, float ldPix, float wvl=1000){
            nRows = r;
            nCols = c;
            nLDPerPix = ldPix;

        }
        
        cv::Mat convertDMToFP(const cv::Mat &dmImShm){
            cv::Mat dmE(dmImShm.rows, dmImShm.cols, CV_32FC2);
            cv::Mat fpE(nRows, nCols, CV_32FC2); //complex float
            cv::Mat fpIm(nRows, nCols, CV_32F); //intensities
            cv::Mat dmIm;
            dmImShm.copyTo(dmIm);
            dmIm = dmIm/wvl; //convert to phase
            
            dmE.forEach<cv::Vec2f>([this, &dmIm](cv::Vec2f &value, const int *position) -> void { 
                    value = cv::Vec2f(std::cos(dmIm.at<float>(position[0], position[1])), std::sin(dmIm.at<float>(position[0], position[1])));

                }); //convert phase to complex E-field
            
            cv::dft(dmE, fpE, cv::DFT_COMPLEX_OUTPUT); //fft
            
            //fix quadrants so 0 freq is at center
            int qx = dmIm.cols/2;
            int qy = dmIm.rows/2;
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

            fpIm = fpIm*1000;
            cv::Mat fpImOut(nRows, nCols, CV_32S);
            fpIm.convertTo(fpImOut, CV_32S);

            return fpImOut;
            
        }

};
