#include "ImageGrabber.h"

ImageGrabber::ImageGrabber(boost::property_tree::ptree &ptree) : mParams(ptree), mIntegrationTime(-1){
    initialize();

}

void ImageGrabber::initialize(){
    BOOST_LOG_TRIVIAL(debug) << "ImageGrabber: opening " << mParams.get<std::string>("name");
    
    int retval = MKIDShmImage_open(&mShmImage, mParams.get<std::string>("name").c_str());
    if(retval != 0){
        BOOST_LOG_TRIVIAL(error) << "Error opening " << mParams.get<std::string>("name");
        throw;

    }
    if(mShmImage.md->nWvlBins != 1){
        BOOST_LOG_TRIVIAL(error) << "Multiple wavelengths " << mShmImage.md->nWvlBins << " not supported";
        throw;

    }

    mRawImageShm = cv::Mat(mShmImage.md->nRows, mShmImage.md->nCols, CV_32S, mShmImage.image);
    BOOST_LOG_TRIVIAL(trace) << "ImageGrabber: done opening " << mParams.get<std::string>("name");

    MKIDShmImage_setWvlRange(&mShmImage, mParams.get("wvlStart", 700), mParams.get("wvlStop", 1400));
    mShmImage.md->useWvl = mParams.get("useWvl", 0);

    //TODO: consider changing centers to floats
    mXCenter = (int)std::round(mParams.get<double>("xCenter"));
    mYCenter = (int)std::round(mParams.get<double>("yCenter"));
    setCtrlRegion();

    if(mParams.get<bool>("useBadPixMask"))
    {
        mBadPixBuff = new char[2*mShmImage.md->nCols*mShmImage.md->nRows];
        loadBadPixMask();

    }

    //initialize bad pixel mask to 0
    else
    {
        mBadPixMask = cv::Mat(mShmImage.md->nRows, mShmImage.md->nCols, CV_16UC1, cv::Scalar(0));
        mBadPixMaskCtrl = cv::Mat(mBadPixMask, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));

    }

    if(mParams.get<bool>("useFlatCal"))
    {
        mFlatCalBuff = new char[8*mShmImage.md->nCols*mShmImage.md->nRows]; //pack flat cal into 64 bit double array
        loadFlatCal();

    }
    
    if(mParams.get<bool>("useDarkSub"))
    {
        mDarkSubBuff = new char[2*mShmImage.md->nCols*mShmImage.md->nRows]; //pack dark into ushort array
        loadDarkSub();

    }

    BOOST_LOG_TRIVIAL(debug) << "ImageGrabber: done initializing";

}

ImageGrabber::ImageGrabber(const ImageGrabber &other){
    close();
    boost::property_tree::ptree ptree = other.getCfgParams();
    mParams = ptree;
    initialize();


}

ImageGrabber &ImageGrabber::operator=(const ImageGrabber &rhs){
    if(this != &rhs){
        close();
        boost::property_tree::ptree ptree = rhs.getCfgParams();
        mParams = ptree;
        initialize();


    }

    return *this;

}


void ImageGrabber::processCtrlRegion()
{
    if(mParams.get<bool>("useDarkSub"))
        applyDarkSubCtrlRegion();
    if(mParams.get<bool>("useFlatCal"))
        applyFlatCalCtrlRegion();

}

void ImageGrabber::processFullImage()
{
    if(mParams.get<bool>("useDarkSub"))
        applyDarkSub();
    if(mParams.get<bool>("useFlatCal"))
        applyFlatCal();
    mCtrlRegionImage = cv::Mat(mRawImageShm, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));

}

void ImageGrabber::startIntegrating(uint64_t startts, double integrationTime)
{
    BOOST_LOG_TRIVIAL(debug) << "ImageGrabber: starting integration";
    MKIDShmImage_startIntegration(&mShmImage, startts, (uint64_t)integrationTime*2);
    mIntegrationTime = integrationTime;
    mUpToDate = false;

}

void ImageGrabber::startIntegrating(uint64_t startts, double integrationTime, int wvlStart, int wvlStop)
{
    BOOST_LOG_TRIVIAL(debug) << "ImageGrabber: starting integration";
    MKIDShmImage_setWvlRange(&mShmImage, wvlStart, wvlStop);
    MKIDShmImage_startIntegration(&mShmImage, startts, (uint64_t)integrationTime*2);
    mIntegrationTime = integrationTime;
    mUpToDate = false;

}

cv::Mat &ImageGrabber::getCtrlRegionImage(bool process) 
{
    BOOST_LOG_TRIVIAL(trace) << "ImageGrabber: waiting...";
    if(!mUpToDate){
        MKIDShmImage_wait(&mShmImage, DONE_SEM_IND);
        mUpToDate = true;

    }
    BOOST_LOG_TRIVIAL(trace) << "ImageGrabber: grabbing ctrl region...";
    mCtrlRegionImage = cv::Mat(mRawImageShm, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));
    if(process)
        processCtrlRegion();
    return mCtrlRegionImage;

}


cv::Mat &ImageGrabber::getImage(bool process)
{
    BOOST_LOG_TRIVIAL(trace) << "ImageGrabber: waiting...";
    if(!mUpToDate){
        MKIDShmImage_wait(&mShmImage, DONE_SEM_IND);
        mUpToDate = true;

    }

    if(process)
        processFullImage();
    return mRawImageShm;

}

const cv::Mat &ImageGrabber::getBadPixMask() const
{
    return mBadPixMask;

}

const cv::Mat &ImageGrabber::getBadPixMaskCtrl() const
{
    return mBadPixMaskCtrl;

}

boost::property_tree::ptree ImageGrabber::getCfgParams() const{
    return mParams;

}

void ImageGrabber::setCtrlRegion()
{
    mXCtrlStart = mXCenter + mParams.get<int>("xCtrlStart");
    mXCtrlEnd = mXCenter + mParams.get<int>("xCtrlEnd");
    mYCtrlStart = mYCenter + mParams.get<int>("yCtrlStart");
    mYCtrlEnd = mYCenter + mParams.get<int>("yCtrlEnd");

    BOOST_LOG_TRIVIAL(debug) << "ImageGrabber: set control region to: start = (" <<
        mXCtrlStart << ", " << mYCtrlStart << "), end = (" << mXCtrlEnd << ", " << mYCtrlEnd << ")";


}

void ImageGrabber::changeCenter(int xCent, int yCent)
{
    mXCenter = xCent;
    mYCenter = yCent;
    setCtrlRegion();
    if(mParams.get<bool>("useBadPixMask"))
        mBadPixMaskCtrl = cv::Mat(mBadPixMask, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));


}

void ImageGrabber::loadBadPixMask()
{
    std::string badPixFn = mParams.get<std::string>("badPixMaskFile");
    std::ifstream badPixFile(badPixFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!badPixFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find bad pixel mask";
    badPixFile.read(mBadPixBuff, 2*mShmImage.md->nCols*mShmImage.md->nRows);
    mBadPixMask = cv::Mat(mShmImage.md->nRows, mShmImage.md->nCols, CV_16UC1, mBadPixBuff);
    mBadPixMaskCtrl = cv::Mat(mBadPixMask, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));

}

void ImageGrabber::loadFlatCal()
{
    std::string flatCalFn = mParams.get<std::string>("flatCalFile");
    std::ifstream flatCalFile(flatCalFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!flatCalFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find flat cal";
    flatCalFile.read(mFlatCalBuff, 8*mShmImage.md->nCols*mShmImage.md->nRows);
    mFlatWeights = cv::Mat(mShmImage.md->nRows, mShmImage.md->nCols, CV_64FC1, mFlatCalBuff);
    mFlatWeightsCtrl = cv::Mat(mFlatWeights, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));
    //cv::imshow("flat", mFlatWeights);
    //cv::waitKey(0);
    //std::cout << mFlatWeights << std::endl;

}

void ImageGrabber::loadDarkSub()
{
    std::string mDarkSubFn = mParams.get<std::string>("mDarkSubFile");
    std::ifstream mDarkSubFile(mDarkSubFn.c_str(), std::ifstream::in|std::ifstream::binary);
    if(!mDarkSubFile.good()) BOOST_LOG_TRIVIAL(warning) << "Could not find dark sub";
    mDarkSubFile.read(mDarkSubBuff, 2*mShmImage.md->nCols*mShmImage.md->nRows);
    mDarkSub = cv::Mat(mShmImage.md->nRows, mShmImage.md->nCols, CV_16UC1, mDarkSubBuff);
    mDarkSubCtrl = cv::Mat(mDarkSub, cv::Range(mYCtrlStart, mYCtrlEnd), cv::Range(mXCtrlStart, mXCtrlEnd));
    mDarkSubCtrl.convertTo(mDarkSubCtrl, CV_64FC1);
    mDarkSub.convertTo(mDarkSub, CV_64FC1);

}

void ImageGrabber::badPixFiltCtrlRegion()
{
    cv::Mat mBadPixMaskCtrlInv = (~mBadPixMaskCtrl)&1;
    cv::Mat ctrlRegionFilt = mCtrlRegionImage.clone();
    cv::medianBlur(mCtrlRegionImage.mul(mBadPixMaskCtrlInv), ctrlRegionFilt, mParams.get<int>("badPixFiltSize"));
    mCtrlRegionImage = ctrlRegionFilt.mul(mBadPixMaskCtrl) + mCtrlRegionImage.mul(mBadPixMaskCtrlInv);

}


void ImageGrabber::applyFlatCal()
{
    BOOST_LOG_TRIVIAL(debug)<<"applying flat";
    mRawImageShm = mRawImageShm.mul(mFlatWeightsCtrl);
    //std::cout << mCtrlRegionImage << std::endl;

}
     

void ImageGrabber::applyFlatCalCtrlRegion()
{
    BOOST_LOG_TRIVIAL(debug)<<"applying flat to control region";
    mCtrlRegionImage = mCtrlRegionImage.mul(mFlatWeightsCtrl);
    //std::cout << mCtrlRegionImage << std::endl;

}

void ImageGrabber::applyDarkSub()
{
    mRawImageShm = mRawImageShm - mDarkSub*mParams.get<double>("integrationTime")/1000;

}

void ImageGrabber::applyDarkSubCtrlRegion()
{
    mCtrlRegionImage = mCtrlRegionImage - mDarkSubCtrl*mParams.get<double>("integrationTime")/1000;

}

int ImageGrabber::getCtrlRegionCounts(){
    return (int)cv::sum(mCtrlRegionImage)[0];

}

void ImageGrabber::displayImage(bool makePlot)
{
    //std::cout << "mRawImageShm " << mRawImageShm << std::endl;
    //cv::namedWindow("DARKNESS Sim Image", cv::WINDOW_NORMAL);
    //std::cout << "mBadPixMaskCtrl" << mBadPixMaskCtrl << std::endl;
     //   cv::imshow("DARKNESS Sim Image", 10*mCtrlRegionImage); 
    cv::Mat ctrlRegionDispImage;
    mCtrlRegionImage.convertTo(ctrlRegionDispImage, CV_16UC1);
    std::cout << ctrlRegionDispImage << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "ctrl region size " << ctrlRegionDispImage.size;

    
    if(makePlot)
    {
        double maxVal;
        double minVal;
        cv::minMaxLoc(mCtrlRegionImage, &minVal, &maxVal);
        cv::namedWindow("DARKNESS Image", cv::WINDOW_NORMAL);
        cv::imshow("DARKNESS Image", mCtrlRegionImage/maxVal);
        //std::cout << mRawImageShm << std::endl;
        cv::waitKey(0);

    }

}


void ImageGrabber::close()
{
    MKIDShmImage_close(&mShmImage);
    if(mParams.get<bool>("useBadPixMask"))
        free(mBadPixBuff);
    if(mParams.get<bool>("useFlatCal"))
        free(mFlatCalBuff);
    if(mParams.get<bool>("useDarkSub"))
    free(mDarkSubBuff);

}

ImageGrabber::~ImageGrabber(){
    close();

}
