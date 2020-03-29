#include "SpeckleToDM.h"

SpeckleToDM::SpeckleToDM(const char dmChanName[80], bool _usenm) : dmChannel(dmChanName), usenm(_usenm){
    dmXSize = dmChannel.getXSize();
    dmYSize = dmChannel.getYSize();
    fullMapShm = cv::Mat(dmYSize, dmXSize, CV_32F, dmChannel.getBufferPtr<float>());

    tempMap = cv::Mat::zeros(dmYSize, dmXSize, CV_32F); 
    probeMap = cv::Mat(dmYSize, dmXSize, CV_32F);
    nullMap = cv::Mat(dmYSize, dmXSize, CV_32F);
    probeMap.setTo(0);
    nullMap.setTo(0);

}


void SpeckleToDM::addProbeSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    probeMap += generateMapFromSpeckle(kvecs, amp, phase);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding probe speckle with k: " 
        << kvecs << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addProbeSpeckle(double kx, double ky, double amp, double phase)
{
    probeMap += generateMapFromSpeckle(cv::Point2d(kx, ky), amp, phase);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding probe speckle with k: " 
        << cv::Point2d(kx, ky) << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addNullingSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    nullMap += generateMapFromSpeckle(kvecs, amp, phase);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding nulling speckle with k: " 
        << kvecs << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addNullingSpeckle(double kx, double ky, double amp, double phase)
{
    nullMap += generateMapFromSpeckle(cv::Point2d(kx, ky), amp, phase);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding nulling speckle with k: " 
        << cv::Point2d(kx, ky) << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::clearProbeSpeckles()
{
    probeMap.setTo(0);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Clearing probe speckles";

}

void SpeckleToDM::clearNullingSpeckles()
{
    nullMap.setTo(0);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Clearing nulling speckles";

}

void SpeckleToDM::updateDM()
{
    cv::add(probeMap, nullMap, fullMapShm);
    dmChannel.postAllSemaphores();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Updating DM with new speckles";

}


cv::Mat SpeckleToDM::generateMapFromSpeckle(const cv::Point2d kvecs, double amp, double phase)
{
    if(usenm)
        amp /= 1000; // dm channel is in um stroke
    float phx, phy;
    tempMap.setTo(0);
    tempMap.forEach<Pixel>([this, &phx, &phy, amp, phase, &kvecs](Pixel &value, const int *position) -> void
        { phy = (double)(((1.0/this->dmYSize)*position[0]-0.5)*kvecs.y);
          phx = (double)(((1.0/this->dmXSize)*position[1]-0.5)*kvecs.x);
          value = (float)amp*std::cos(phx + phy + phase);

        });

    //int r, c;

    //for(r=0; r<dmYSize; r++)
    //    for(c=0; c<dmXSize; c++){
    //        phy = (double)(((1.0/this->dmYSize)*r-0.5)*kvecs.y);
    //        phx = (double)(((1.0/this->dmXSize)*c-0.5)*kvecs.x);
    //        tempMap.at<Pixel>(r, c) = (float)amp*std::cos(phx + phy + phase);

    //    }
    
    
    return tempMap;

}

int SpeckleToDM::getXSize(){ return dmXSize;}
int SpeckleToDM::getYSize(){ return dmYSize;}
