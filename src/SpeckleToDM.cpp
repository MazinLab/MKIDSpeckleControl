#include "SpeckleToDM.h"

SpeckleToDM::SpeckleToDM(const char dmChanName[80], bool _usenm) : dmChannel(dmChanName), usenm(_usenm){
    if((dmChannel.getXSize() != DM_X_SIZE) or (dmChannel.getYSize() != DM_X_SIZE))
        throw;

    fullMapShm = dmChannel.getBufferPtr<float>();


}

void setMapToZero(float *map){
    memset(map, 0, DM_X_SIZE*DM_Y_SIZE*sizeof(float));

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
    setMapToZero(probeMap);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Clearing probe speckles";

}

void SpeckleToDM::clearNullingSpeckles()
{
    setMapToZero(nullMap);
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Clearing nulling speckles";

}

void SpeckleToDM::updateDM()
{
    int r, c;
    for(r=0; r<DM_Y_SIZE; r++)
        for(c=0; c<DM_X_SIZE; c++)
            fullMapShm[r*DM_X_SIZE + c] = probeMap[r*DM_X_SIZE + c] + nullMap[r*DM_X_SIZE + c];
    dmChannel.postAllSemaphores();
    BOOST_LOG_TRIVIAL(debug) << "SpeckleToDM " << dmChannel.getName() << ": Updating DM with new speckles";

}


void SpeckleToDM::generateMapFromSpeckle(const cv::Point2d kvecs, double amp, double phase, float *map)
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

    for(r=0; r<DM_Y_SIZE; r++)
        for(c=0; c<DM_X_SIZE; c++){
            phy = (float)(((1.0/DM_Y_SIZE)*r-0.5)*kvecs.y);
            phx = (float)(((1.0/DM_X_SIZE)*c-0.5)*kvecs.x);
            *(map + r*DM_X_SIZE + c) += (float)amp*std::cos(phx + phy + phase);

        }
    
    

}

int SpeckleToDM::getXSize(){ return DM_X_SIZE;}
int SpeckleToDM::getYSize(){ return DM_Y_SIZE;}
