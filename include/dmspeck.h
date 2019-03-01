#ifndef DMSPECK_H
#define DMSPECK_H
/*
 * Simple wrapper for speckles on the DM. 
 */
typedef struct
{
    double kx;
    double ky;
    double amp;
    double phase;
    bool isNull;

} dmspeck;

#endif
