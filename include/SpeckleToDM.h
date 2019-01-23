#include <iostream>

#include "dmChannel.h"

typedef struct
{
    double kx;
    double ky;
    double amp;

} dmspeck;

class SpeckleToDM
{
    private:
        std::vector<dmspeck> probeSpeckles;
        std::vector<dmspeck> nullingSpeckles;
        cv::Mat probeMap;
        cv::Mat nullMap;

    public:
        void addProbeSpeckle(double kx, double ky, double amp);
        void addNullingSpeckle(double kx, double ky, double amp);
        void updateCalParams(int placeholder);

}
