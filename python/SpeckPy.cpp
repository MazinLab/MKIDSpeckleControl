#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/overloads.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include "SpeckleNuller.h"
#include "PTreeWrapper.h"
#include "loopFunctions.h"
#include "SpeckleToDM.h"

namespace bp = boost::python;

boost::property_tree::ptree extractPropertyTree(bp::object pytree){
    PTreeWrapper &pwrapper = bp::extract<PTreeWrapper&>(pytree);
    return pwrapper.getCXXObject();

}

bp::list run(int nIters, bp::object pyparams, bool returnLC=false){
    boost::property_tree::ptree params = extractPropertyTree(pyparams);
    //std::cout << params.get<std::string>("ImgParams.name") << std::endl;
    std::vector<int> lightCurve = loopfunctions::runLoop(nIters, params, returnLC);

    bp::list lcList;

    for(int counts: lightCurve)
        lcList.append(counts);

    return lcList; 

}

bp::list runNoLC(int nIters, bp::object pyparams){
    return run(nIters, pyparams, false);}

void setTraceLog(){
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);

}

void setDebugLog(){
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

}

void setInfoLog(){
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);

}

void setWarningLog(){
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);

}

BOOST_PYTHON_MODULE(speckpy){
    bp::class_<SpeckleToDM>("SpeckleToDM", bp::init<const char*>())
        .def("addProbeSpeckle", static_cast<void(SpeckleToDM::*)(double, double, double, double)>(&SpeckleToDM::addProbeSpeckle))
        .def("addNullingSpeckle", static_cast<void(SpeckleToDM::*)(double, double, double, double)>(&SpeckleToDM::addNullingSpeckle))
        //.def("addProbeSpeckle", &SpeckleToDM::addProbeSpeckle, addProbeSpeckle_overloads())
        //.def("addNullingSpeckle", addNullingSpeckle)
        .def("clearProbeSpeckles", &SpeckleToDM::clearProbeSpeckles)
        .def("clearNullingSpeckles", &SpeckleToDM::clearNullingSpeckles)
        .def("updateDM", &SpeckleToDM::updateDM)
        .def("getXSize", &SpeckleToDM::getXSize)
        .def("getYSize", &SpeckleToDM::getYSize);

    bp::class_<PTreeWrapper>("PropertyTree")
        .def("read_info", &PTreeWrapper::read_info)
        .def("get", &PTreeWrapper::get)
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const std::string &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const double &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const int &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const bool &)>(&PTreeWrapper::add))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const std::string &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const double &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const int &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const bool &)>(&PTreeWrapper::put))
        .def("write", &PTreeWrapper::write);

    bp::def("runLoop", &run);
    bp::def("runLoop", &runNoLC);
    bp::def("setTraceLog", &setTraceLog);
    bp::def("setDebugLog", &setDebugLog);
    bp::def("setInfoLog", &setInfoLog);
    bp::def("setWarningLog", &setWarningLog);

}
