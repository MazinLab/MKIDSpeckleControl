#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/ptree.hpp>

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#ifndef DMCHANNEL_H
#define DMCHANNEL_H
/*
 * Wrapper around CACAO IMAGE struct; intended
 * for writing to a single shared memory DM channel. Assumes
 * DM channel was already created by running CACAO process.
 * Typical usage:
 *  1. Open DM channel (constructor)
 *  2. Write image to data buffer (given by getBufferPtr)
 *  3. Post semaphores to tell CACAO to update DM
 */
class DMChannel
{
    private:
        IMAGE dmImage;
        bool isOpen;

        void initializeDMShm(const char name[80]);

    public:
        
        /*
         * Constructor. 
         * @param name Name of the DM channel to open. NOT full path;
         * (i.e. /tmp/dm00disp00.im.shm would have name=dm00disp00)
         * path is specified by SHAREDMEMDIR in ImageStruct.h
         */
        DMChannel(const char name[80]);
        DMChannel(const DMChannel &chan);
        DMChannel();

        DMChannel& operator=(const DMChannel &rhs);

        /*
         * Returns pointer to data stored in DM channel. T must be
         * the same data type as data in IMAGE struct. See ImageStruct.h
         * for more information.
         */
        template <class T> T *getBufferPtr();
        // void *getBufferPtr();

        /*
         * Posts all (10) semaphores of DM channel
         */
        void postAllSemaphores();

        /*
         * Posts a single semaphore given by index.
         * @param index Index of semaphore to post.
         */
        void postSemaphore(int index);

        // Getters:
        int getXSize() const;
        int getYSize() const;
        std::string getName() const;

        void close();

        void save(const char filename[200]);

        /*
         * Destructor. Closes DM channel.
         */
        ~DMChannel(); //free dmImage


};
#endif
