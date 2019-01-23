#include <cstdio>
#include <cstdlib>
#include <iostream>

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
        char name[80];

    public:
        DMChannel(const char *name);
        template <class T> getBufferPtr();
        // void *getBufferPtr();
        void incrementAllSemaphores();
        void incrementSemaphore();
        ~DMChannel(); //free dmImage


}
#endif
