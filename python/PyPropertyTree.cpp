#include <boost/python.hpp>
#include <fstream>
#include "PTreeWrapper.h"

using namespace boost::python;


BOOST_PYTHON_MODULE(propertytree){
    class_<PTreeWrapper>("PTree")
        .def("read_info", &PTreeWrapper::read_info)
        .def("get", &PTreeWrapper::get)
        .def("getCXXObject", &PTreeWrapper::getCXXObject)
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const std::string &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const double &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const int &)>(&PTreeWrapper::add))
        .def("add", static_cast<void (PTreeWrapper::*)(const std::string &, const bool &)>(&PTreeWrapper::add))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const std::string &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const double &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const int &)>(&PTreeWrapper::put))
        .def("put", static_cast<void (PTreeWrapper::*)(const std::string &, const bool &)>(&PTreeWrapper::put))
        .def("write", &PTreeWrapper::write);

}

