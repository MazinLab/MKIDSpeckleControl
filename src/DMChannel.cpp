#include "DMChannel.h"

DMChannel::DMChannel(const char name[80])
{
    int retVal;
    dmImage = new IMAGE;
    retVal = ImageStreamIO_openIm(dmImage, name);
    if(retVal==0)
        BOOST_LOG_TRIVIAL(info) << "DM Channel " << name << " opened successfully";
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "DM Channel " << name << " open failed. Exiting.";        
        exit(-1);

    }

}

DMChannel::~DMChannel()
{
    ImageStreamIO_closeIm(dmImage);

}
 
