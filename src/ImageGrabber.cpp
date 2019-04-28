#include "ImageGrabber.h"

ImageGrabber::ImageGrabber(boost::property_tree::ptree &ptree) : cfgParams(ptree){
    initialize();

}

void ImageGrabber::initialize(){
    BOOST_LOG_TRIVIAL(debug) << "Opening Image Buffer...";
    
    MKIDShmImage_open(&shmImage, cfgParams.get<std::string>("ImgParams.shmName").c_str());
    if(shmImage.md->nWvlBins != 1){
        BOOST_LOG_TRIVIAL(error) << "Multiple wavelengths not supported";
        throw;

    }

    xCenter = cfgParams.get<int>("ImgParams.xCenter");
    yCenter = cfgParams.get<int>("ImgParams.yCenter");
    setCtrlRegion();

    if(cfgParams.get<bool>("ImgParams.useBadPixMask"))
    {
        badPixArr = new char[2*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows")];
        loadBadPixMask();

    }

    //initialize bad pixel mask to 0
    else
    {
        badPixMask = cv::Mat(cfgParams.get<int>("ImgParams.nRows"), cfgParams.get<int>("ImgParams.nCols"), CV_16UC1, 0);
        badPixMaskCtrl = cv::Mat(badPixMask, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));

    }

    if(cfgParams.get<bool>("ImgParams.useFlatCal"))
    {
        flatCalArr = new char[8*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows")]; //pack flat cal into 64 bit double array
        loadFlatCal();

    }
    
    if(cfgParams.get<bool>("ImgParams.useDarkSub"))
    {
        darkSubArr = new char[2*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows")]; //pack dark into ushort array
        loadDarkSub();

    }

}

ImageGrabber::ImageGrabber(const ImageGrabber &other){
    close();
    boost::property_tree::ptree ptree = other.getCfgParams();
    cfgParams = ptree;
    initialize();


}

ImageGrabber &ImageGrabber::operator=(const ImageGrabber &rhs){
    if(this != &rhs){
        close();
        boost::property_tree::ptree ptree = rhs.getCfgParams();
        cfgParams = ptree;
        initialize();


    }

    return *this;

}


void ImageGrabber::readNextImage()
{
    MKIDShmImage_wait(&shmImage, 0);
    rawImageShm = cv::Mat(cv::Size(cfgParams.get<int>("ImgParams.nCols"), cfgParams.get<int>("ImgParams.nRows")), CV_32S, shmImage.image);
    copyControlRegionFromShm();
    copyFullImageFromShm();

}

void ImageGrabber::processCtrlRegion()
{
    if(cfgParams.get<bool>("ImgParams.useDarkSub"))
        applyDarkSubCtrlRegion();
    if(cfgParams.get<bool>("ImgParams.useFlatCal"))
        applyFlatCalCtrlRegion();

}

void ImageGrabber::processFullImage()
{
    if(cfgParams.get<bool>("ImgParams.useDarkSub"))
        applyDarkSub();
    if(cfgParams.get<bool>("ImgParams.useFlatCal"))
        applyFlatCal();
    ctrlRegionImage = cv::Mat(fullImage, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));

}

void ImageGrabber::startIntegrating(uint64_t startts)
{
    MKIDShmImage_startIntegration(&shmImage, startts, cfgParams.get<int>("ImgParams.integrationTime"));

}

void ImageGrabber::startIntegrating(uint64_t startts, uint64_t integrationTime)
{
    MKIDShmImage_startIntegration(&shmImage, startts, integrationTime);

}

void ImageGrabber::startIntegrating(uint64_t startts, uint64_t integrationTime, int wvlStart, int wvlStop)
{
    MKIDShmImage_setWvlRange(&shmImage, wvlStart, wvlStop);
    MKIDShmImage_startIntegration(&shmImage, startts, integrationTime);

}

void ImageGrabber::grabControlRegion() //DEPRECATED B/C ctrlRegionImage is now a float64
{
    ctrlRegionImage = cv::Mat(rawImageShm, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));

}

void ImageGrabber::copyControlRegionFromShm()
{
    cv::Mat ctrlRegionTmp = cv::Mat(rawImageShm, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));
    ctrlRegionTmp.convertTo(ctrlRegionImage, CV_64FC1);

}

void ImageGrabber::copyFullImageFromShm()
{
    cv::Mat imageTmp = rawImageShm.clone();
    imageTmp.convertTo(fullImage, CV_64FC1);

}

cv::Mat &ImageGrabber::getCtrlRegionImage()
{
    return ctrlRegionImage;

}

cv::Mat &ImageGrabber::getFullImage()
{
    return fullImage;

}

cv::Mat &ImageGrabber::getRawImageShm()
{
    return rawImageShm;

}

cv::Mat &ImageGrabber::getBadPixMask()
{
    return badPixMask;

}

cv::Mat &ImageGrabber::getBadPixMaskCtrl()
{
    return badPixMaskCtrl;

}

void ImageGrabber::setCtrlRegion()
{
    xCtrlStart = xCenter + cfgParams.get<int>("ImgParams.xCtrlStart");
    xCtrlEnd = xCenter + cfgParams.get<int>("ImgParams.xCtrlEnd");
    yCtrlStart = yCenter + cfgParams.get<int>("ImgParams.yCtrlStart");
    yCtrlEnd = yCenter + cfgParams.get<int>("ImgParams.yCtrlEnd");

}

void ImageGrabber::changeCenter(int xCent, int yCent)
{
    xCenter = xCent;
    yCenter = yCent;
    setCtrlRegion();
    if(cfgParams.get<bool>("ImgParams.useBadPixMask"))
        badPixMaskCtrl = cv::Mat(badPixMask, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));


}

void ImageGrabber::loadBadPixMask()
{
    std::string badPixFn = cfgParams.get<std::string>("ImgParams.badPixMaskFile");
    std::ifstream badPixFile(badPixFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!badPixFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find bad pixel mask";
    badPixFile.read(badPixArr, 2*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows"));
    badPixMask = cv::Mat(cfgParams.get<int>("ImgParams.nRows"), cfgParams.get<int>("ImgParams.nCols"), CV_16UC1, badPixArr);
    badPixMaskCtrl = cv::Mat(badPixMask, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));

}

void ImageGrabber::loadFlatCal()
{
    std::string flatCalFn = cfgParams.get<std::string>("ImgParams.flatCalFile");
    std::ifstream flatCalFile(flatCalFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!flatCalFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find flat cal";
    flatCalFile.read(flatCalArr, 8*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows"));
    flatWeights = cv::Mat(cfgParams.get<int>("ImgParams.nRows"), cfgParams.get<int>("ImgParams.nCols"), CV_64FC1, flatCalArr);
    flatWeightsCtrl = cv::Mat(flatWeights, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));
    //cv::imshow("flat", flatWeights);
    //cv::waitKey(0);
    //std::cout << flatWeights << std::endl;

}

void ImageGrabber::loadDarkSub()
{
    std::string darkSubFn = cfgParams.get<std::string>("ImgParams.darkSubFile");
    std::ifstream darkSubFile(darkSubFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!darkSubFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find dark sub";
    darkSubFile.read(darkSubArr, 2*cfgParams.get<int>("ImgParams.nCols")*cfgParams.get<int>("ImgParams.nRows"));
    darkSub = cv::Mat(cfgParams.get<int>("ImgParams.nRows"), cfgParams.get<int>("ImgParams.nCols"), CV_16UC1, darkSubArr);
    darkSubCtrl = cv::Mat(darkSub, cv::Range(yCtrlStart, yCtrlEnd), cv::Range(xCtrlStart, xCtrlEnd));
    darkSubCtrl.convertTo(darkSubCtrl, CV_64FC1);
    darkSub.convertTo(darkSub, CV_64FC1);

}

void ImageGrabber::badPixFiltCtrlRegion()
{
    cv::Mat badPixMaskCtrlInv = (~badPixMaskCtrl)&1;
    cv::Mat ctrlRegionFilt = ctrlRegionImage.clone();
    cv::medianBlur(ctrlRegionImage.mul(badPixMaskCtrlInv), ctrlRegionFilt, cfgParams.get<int>("ImgParams.badPixFiltSize"));
    ctrlRegionImage = ctrlRegionFilt.mul(badPixMaskCtrl) + ctrlRegionImage.mul(badPixMaskCtrlInv);

}


void ImageGrabber::applyFlatCal()
{
    BOOST_LOG_TRIVIAL(debug)<<"applying flat";
    fullImage = fullImage.mul(flatWeightsCtrl);
    //std::cout << ctrlRegionImage << std::endl;

}
     

void ImageGrabber::applyFlatCalCtrlRegion()
{
    BOOST_LOG_TRIVIAL(debug)<<"applying flat to control region";
    ctrlRegionImage = ctrlRegionImage.mul(flatWeightsCtrl);
    //std::cout << ctrlRegionImage << std::endl;

}

void ImageGrabber::applyDarkSub()
{
    fullImage = fullImage - darkSub*cfgParams.get<double>("ImgParams.integrationTime")/1000;

}

void ImageGrabber::applyDarkSubCtrlRegion()
{
    ctrlRegionImage = ctrlRegionImage - darkSubCtrl*cfgParams.get<double>("ImgParams.integrationTime")/1000;

}

void ImageGrabber::displayImage(bool makePlot)
{
    //std::cout << "rawImageShm " << rawImageShm << std::endl;
    //cv::namedWindow("DARKNESS Sim Image", cv::WINDOW_NORMAL);
    //std::cout << "badPixMaskCtrl" << badPixMaskCtrl << std::endl;
     //   cv::imshow("DARKNESS Sim Image", 10*ctrlRegionImage); 
    cv::Mat ctrlRegionDispImage;
    ctrlRegionImage.convertTo(ctrlRegionDispImage, CV_16UC1);
    std::cout << ctrlRegionDispImage << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "ctrl region size " << ctrlRegionDispImage.size;

    
    if(makePlot)
    {
        double maxVal;
        double minVal;
        cv::minMaxLoc(ctrlRegionImage, &minVal, &maxVal);
        cv::namedWindow("DARKNESS Image", cv::WINDOW_NORMAL);
        cv::imshow("DARKNESS Image", ctrlRegionImage/maxVal);
        //std::cout << rawImageShm << std::endl;
        cv::waitKey(0);

    }

}


void ImageGrabber::close()
{
    MKIDShmImage_close(&shmImage);
    free(badPixArr);
    free(flatCalArr);
    free(darkSubArr);

}

ImageGrabber::~ImageGrabber(){
    close();

}
