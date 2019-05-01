#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/overloads.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include "SpeckleNuller.h"
#include "PTreeWrapper.h"
#include "loopFunctions.h"
#include "SpeckleToDM.h"

using namespace boost::python;

boost::property_tree::ptree extractPropertyTree(object pytree){
    PTreeWrapper &pwrapper = extract<PTreeWrapper&>(pytree);
    return pwrapper.getCXXObject();

}

void run(int nIters, object pyparams){
    boost::property_tree::ptree params = extractPropertyTree(pyparams);
    std::cout << params.get<std::string>("ImgParams.name") << std::endl;

}

BOOST_PYTHON_MODULE(speckpy){
    class_<SpeckleToDM>("SpeckleToDM", init<const char*>())
        .def("addProbeSpeckle", static_cast<void(SpeckleToDM::*)(double, double, double, double)>(&SpeckleToDM::addProbeSpeckle))
        .def("addNullingSpeckle", static_cast<void(SpeckleToDM::*)(double, double, double, double)>(&SpeckleToDM::addNullingSpeckle))
        //.def("addProbeSpeckle", &SpeckleToDM::addProbeSpeckle, addProbeSpeckle_overloads())
        //.def("addNullingSpeckle", addNullingSpeckle)
        .def("clearProbeSpeckles", &SpeckleToDM::clearProbeSpeckles)
        .def("clearNullingSpeckles", &SpeckleToDM::clearNullingSpeckles)
        .def("updateDM", &SpeckleToDM::updateDM)
        .def("getXSize", &SpeckleToDM::getXSize)
        .def("getYSize", &SpeckleToDM::getYSize);

    class_<PTreeWrapper>("PTree")
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

    def("runLoop", &run);

}
