#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <boost/log/trivial.hpp>

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#ifndef DMCHANNEL_H
#define DMCHANNEL_H
/*
 * Wrapper around CACAO IMAGE struct; intended
 * for writing to shared memory DM channels. Assumes
 * DM channel was already created by running CACAO process
 */
class DMChannel
{
    private:
        IMAGE* dmImage;

    public:
        DMChannel(const char name[80]);
        template <class T> T *getBufferPtr();
        // void *getBufferPtr();
        void incrementAllSemaphores();
        void incrementSemaphore(int index);
        ~DMChannel(); //free dmImage


};
#endif
