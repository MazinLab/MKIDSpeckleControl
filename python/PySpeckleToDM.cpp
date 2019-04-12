#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/overloads.hpp>
#include "SpeckleToDM.h"

using namespace boost::python;

//void (SpeckleToDM::*addProbeSpeckle)(double, double, double, double) = &SpeckleToDM::addProbeSpeckle;
//void (SpeckleToDM::*addNullingSpeckle)(double, double, double, double) = &SpeckleToDM::addNullingSpeckle;

BOOST_PYTHON_MODULE(speckletodm){
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

}
