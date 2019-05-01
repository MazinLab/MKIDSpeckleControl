#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/overloads.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include "SpeckleNuller.h"
#include "PTreeWrapper.h"
#include "loopFunctions.h"

using namespace boost::python;

void run(int nIters, object pyparams){
    PTreeWrapper &pwrapper = extract<PTreeWrapper&>(pyparams);
    boost::property_tree::ptree params = pwrapper.getCXXObject();
    std::cout << params.get<std::string>("ImgParams.name") << std::endl;

}

BOOST_PYTHON_MODULE(specklenuller){
    def("run", &run);

}
