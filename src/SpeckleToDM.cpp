#include "SpeckleToDM.h"

SpeckleToDM::SpeckleToDM(const char dmChanName[80], boost::property_tree::ptree &ptree)
{
    cfgParams = ptree;
    dmChannel = new DMChannel(dmChanName);
    dmXSize = (*dmChannel).getXSize();
    dmYSize = (*dmChannel).getYSize();
    fullMapShm = cv::Mat(dmYSize, dmXSize, CV_32F, (*dmChannel).getBufferPtr<float>());
    
    probeMap = cv::Mat(dmYSize, dmXSize, CV_32F);
    nullMap = cv::Mat(dmYSize, dmXSize, CV_32F);

}

SpeckleToDM::SpeckleToDM(const char dmChanName[80])
{
    dmChannel = new DMChannel(dmChanName);
    dmXSize = (*dmChannel).getXSize();
    dmYSize = (*dmChannel).getYSize();
    fullMapShm = cv::Mat(dmYSize, dmXSize, CV_32F, (*dmChannel).getBufferPtr<float>());
    
    probeMap = cv::Mat(dmYSize, dmXSize, CV_32F);
    nullMap = cv::Mat(dmYSize, dmXSize, CV_32F);

}

void SpeckleToDM::addProbeSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    probeMap += generateMapFromSpeckle(kvecs, amp, phase);

}

void SpeckleToDM::addNullingSpeckle(cv::Point2d kvecs, double amp, double phase)
{
    nullMap += generateMapFromSpeckle(kvecs, amp, phase);

}

void SpeckleToDM::clearProbeSpeckles()
{
    probeMap.setTo(0);

}

void SpeckleToDM::clearNullingSpeckles()
{
    nullMap.setTo(0);

}

void SpeckleToDM::updateDM()
{
    cv::add(probeMap, nullMap, fullMapShm);
    (*dmChannel).postAllSemaphores();

}


cv::Mat SpeckleToDM::generateMapFromSpeckle(const cv::Point2d kvecs, double amp, double phase)
{
    float phx, phy;
    cv::Mat map = cv::Mat::zeros(dmYSize, dmXSize, CV_32F);
    map.forEach<Pixel>([this, &phx, &phy, amp, phase, &kvecs](Pixel &value, const int *position) -> void
        { phy = (double)(((1.0/this->dmYSize)*position[0]-0.5)*kvecs.y);
          phx = (double)(((1.0/this->dmXSize)*position[1]-0.5)*kvecs.x);
          value = (float)amp*std::cos(phx + phy + phase);

        });
    
    
    return map;

}

int SpeckleToDM::getXSize(){ return dmXSize;}
int SpeckleToDM::getYSize(){ return dmYSize;}
