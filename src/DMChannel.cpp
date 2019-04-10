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

DMChannel::DMChannel(){;} //placeholder

void DMChannel::postAllSemaphores()
{
    ImageStreamIO_sempost(dmImage, -1);

}


//TODO: implement the rest of the datatypes
template <class T> T* DMChannel::getBufferPtr(){;}

template <> float* DMChannel::getBufferPtr<float>()
{
    if((dmImage->md)->datatype != _DATATYPE_FLOAT)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        exit(-1);
    
    }

    return dmImage->array.F;

}

template <> double* DMChannel::getBufferPtr<double>()
{
    if((dmImage->md)->datatype != _DATATYPE_DOUBLE)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        exit(-1);
    
    }

    return dmImage->array.D;

}

template <> uint8_t* DMChannel::getBufferPtr<uint8_t>()
{
    if((dmImage->md)->datatype != _DATATYPE_UINT8)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        exit(-1);
    
    }
            
    return dmImage->array.UI8;

}

int DMChannel::getXSize(){ return (dmImage->md)->size[0];}
int DMChannel::getYSize(){ return (dmImage->md)->size[1];}
std::string DMChannel::getName(){ return (std::string)dmImage->name;}

DMChannel::~DMChannel()
{
    ImageStreamIO_closeIm(dmImage);
    BOOST_LOG_TRIVIAL(info) << "DM Channel " << dmImage->name << " closed";

}

