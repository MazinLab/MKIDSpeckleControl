#include "SpeckleToDM.h"

SpeckleToDM::SpeckleToDM(const char dmChanName[80], bool _usenm) : dmChannel(dmChanName), usenm(_usenm){
    dmXSize = dmChannel.getXSize();
    dmYSize = dmChannel.getYSize();
    if((dmXSize != DM_X_SIZE) or (dmXSize != DM_X_SIZE))
        throw;

    fullMapShm = cv::Mat(dmYSize, dmXSize, CV_32F, dmChannel.getBufferPtr<float>());

    tempMap = cv::Mat(dmYSize, dmXSize, CV_32F, tempMapBuf); 
    probeMap = cv::Mat(dmYSize, dmXSize, CV_32F, probeMapBuf);
    nullMap = cv::Mat(dmYSize, dmXSize, CV_32F, nullMapBuf);
    probeMap.setTo(0);
    nullMap.setTo(0);

}


void SpeckleToDM::addProbeSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    generateMapFromSpeckle(kvecs, amp, phase, probeMap);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding probe speckle with k: " 
        << kvecs << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addProbeSpeckle(double kx, double ky, double amp, double phase)
{
    generateMapFromSpeckle(cv::Point2d(kx, ky), amp, phase, probeMap);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding probe speckle with k: " 
        << cv::Point2d(kx, ky) << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addNullingSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    generateMapFromSpeckle(kvecs, amp, phase, nullMap);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Adding nulling speckle with k: " 
        << kvecs << ", amplitude: " << amp << ", phase: " << phase;

}

void SpeckleToDM::addNullingSpeckle(double kx, double ky, double amp, double phase)
{
    generateMapFromSpeckle(cv::Point2d(kx, ky), amp, phase, nullMap);
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


void SpeckleToDM::generateMapFromSpeckle(const cv::Point2d kvecs, double amp, double phase, cv::Mat &map)
{
    if(usenm)
        amp /= 1000; // dm channel is in um stroke
    float phx, phy;
    //tempMap.forEach<Pixel>([this, &phx, &phy, amp, phase, &kvecs](Pixel &value, const int *position) -> void
    //    { phy = (double)(((1.0/this->dmYSize)*position[0]-0.5)*kvecs.y);
    //      phx = (double)(((1.0/this->dmXSize)*position[1]-0.5)*kvecs.x);
    //      value = (float)amp*std::cos(phx + phy + phase);

    //    });

    int r, c;
    float *mapPtr = map.ptr<float>(0);

    for(r=0; r<dmYSize; r++)
        for(c=0; c<dmXSize; c++){
            phy = (float)(((1.0/this->dmYSize)*r-0.5)*kvecs.y);
            phx = (float)(((1.0/this->dmXSize)*c-0.5)*kvecs.x);
            *(mapPtr + r*dmXSize + c) += (float)amp*std::cos(phx + phy + phase);

        }
    
    

}

int SpeckleToDM::getXSize(){ return dmXSize;}
int SpeckleToDM::getYSize(){ return dmYSize;}
