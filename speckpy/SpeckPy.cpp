#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/overloads.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <boost/log/trivial.hpp>
#include <boost/log/common.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include "SpeckleNuller.h"
#include "PTreeWrapper.h"
#include "loopFunctions.h"
#include "SpeckleToDM.h"

namespace bp = boost::python;

boost::property_tree::ptree extractPropertyTree(bp::object pytree){
    PTreeWrapper &pwrapper = bp::extract<PTreeWrapper&>(pytree);
    return pwrapper.getCXXObject();

}

bp::list run(int nIters, bp::object pyparams, bool returnLC=false, bool useAbsTiming=false){
    boost::property_tree::ptree params = extractPropertyTree(pyparams);
    //std::cout << params.get<std::string>("ImgParams.name") << std::endl;
    std::vector<int> lightCurve = loopfunctions::runLoop(nIters, params, returnLC, useAbsTiming);

    bp::list lcList;

    for(int counts: lightCurve)
        lcList.append(counts);

    return lcList; 

}

bp::list runNoLC(int nIters, bp::object pyparams){
    return run(nIters, pyparams, false, false);}

bp::list runRelTiming(int nIters, bp::object pyparams, bool returnLC){
    return run(nIters, pyparams, returnLC, false);}

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

void addLogfile(const std::string &logfile){
    boost::log::core::get()->remove_all_sinks();
    boost::log::add_common_attributes();
    boost::log::add_file_log(
            boost::log::keywords::file_name = logfile,
            boost::log::keywords::format = 
                (boost::log::expressions::stream << "[" <<
                    boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
                    << "][" << boost::log::trivial::severity << "]: " << boost::log::expressions::smessage)
            
            );

    boost::log::add_console_log(std::cout,  
            boost::log::keywords::format = 
                (boost::log::expressions::stream << "[" <<
                    boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
                    << "][" << boost::log::trivial::severity << "]: " << boost::log::expressions::smessage)
            );

}

BOOST_PYTHON_MODULE(_speckpy){
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
    bp::def("runLoop", &runRelTiming);
    bp::def("setTraceLog", &setTraceLog);
    bp::def("setDebugLog", &setDebugLog);
    bp::def("setInfoLog", &setInfoLog);
    bp::def("setWarningLog", &setWarningLog);
    bp::def("addLogfile", &addLogfile);

}
