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


//TODO: implement the rest of the datatypes
template <class T> T* DMChannel::getBufferPtr(){;}

template <> float* DMChannel::getBufferPtr<float>()
{
    if((dmImage->md)->atype != _DATATYPE_FLOAT)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        exit(-1);
    
    }

    return dmImage->array.F;

}

template <> uint8_t* DMChannel::getBufferPtr<uint8_t>()
{
    if((dmImage->md)->atype != _DATATYPE_UINT8)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        exit(-1);
    
    }
            
    return dmImage->array.UI8;

}
