#include "DMChannel.h"

DMChannel::DMChannel(const char name[80])
{
    isOpen = false;
    initializeDMShm(name);

}

DMChannel::DMChannel(){
    isOpen = false;

}

DMChannel::DMChannel(const DMChannel &chan){
    isOpen = false;
    initializeDMShm(chan.getName().c_str());

}

DMChannel &DMChannel::operator=(const DMChannel &rhs){
    if(this != &rhs){
        if(isOpen)
            close();

        initializeDMShm(rhs.getName().c_str());


    }

    return *this;

}

void DMChannel::initializeDMShm(const char name[80]){
    int retVal;
    retVal = ImageStreamIO_openIm(&dmImage, name);
    if(retVal==0)
        BOOST_LOG_TRIVIAL(info) << "DM Channel " << name << " opened successfully";
    else
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel " << name << " open failed. Exiting.";        
        throw;

    }

    isOpen = true;

}


void DMChannel::postAllSemaphores()
{
    //dmImage->md->write = 0;
    dmImage.md->cnt0++;
    ImageStreamIO_sempost(&dmImage, -1);

}


//TODO: implement the rest of the datatypes
template <class T> T* DMChannel::getBufferPtr(){;}

template <> float* DMChannel::getBufferPtr<float>()
{
    if((dmImage.md)->datatype != _DATATYPE_FLOAT)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        throw;
    
    }

    return dmImage.array.F;

}

template <> double* DMChannel::getBufferPtr<double>()
{
    if((dmImage.md)->datatype != _DATATYPE_DOUBLE)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        throw;
    
    }

    return dmImage.array.D;

}

template <> uint8_t* DMChannel::getBufferPtr<uint8_t>()
{
    if((dmImage.md)->datatype != _DATATYPE_UINT8)
    {
        BOOST_LOG_TRIVIAL(error) << "DM Channel type mismatch!";        
        throw;
    
    }
            
    return dmImage.array.UI8;

}

int DMChannel::getXSize() const{ return (dmImage.md)->size[0];}
int DMChannel::getYSize() const{ return (dmImage.md)->size[1];}
std::string DMChannel::getName() const{ return (std::string)dmImage.name;}

void DMChannel::close(){
    ImageStreamIO_closeIm(&dmImage);
    isOpen = false;
    BOOST_LOG_TRIVIAL(info) << "DM Channel " << dmImage.name << " closed";

}


DMChannel::~DMChannel()
{
    close();

}

