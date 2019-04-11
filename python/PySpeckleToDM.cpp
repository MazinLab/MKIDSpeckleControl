#include <boost/python.hpp>
#include "SpeckleToDM.h"

using namespace boost::python;

void (SpeckleToDM::*addProbeSpeckle)(double, double, double, double) = &SpeckleToDM::addProbeSpeckle;
void (SpeckleToDM::*addNullingSpeckle)(double, double, double, double) = &SpeckleToDM::addNullingSpeckle;

BOOST_PYTHON_MODULE(speckletodm){
        class_<SpeckleToDM>("SpeckleToDM", init<const char*>())
            .def("addProbeSpeckle", addProbeSpeckle)
            .def("addNullingSpeckle", addNullingSpeckle)
            .def("clearProbeSpeckles", &SpeckleToDM::clearProbeSpeckles)
            .def("clearNullingSpeckles", &SpeckleToDM::clearNullingSpeckles)
            .def("updateDM", &SpeckleToDM::updateDM)
            .def("getXSize", &SpeckleToDM::getXSize)
            .def("getYSize", &SpeckleToDM::getYSize);

}
