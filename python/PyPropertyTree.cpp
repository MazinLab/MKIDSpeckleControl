#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/python.hpp>
#include <fstream>

using namespace boost::python;


class PTreeWrapper{
    boost::property_tree::ptree myTree;
    
    public:
        void read_info(const std::string &filename){
            boost::property_tree::info_parser::read_info(filename, myTree);

        }

        std::string get(const std::string &key){
            return myTree.get<std::string>(key);

        }

        void add(const std::string &key, const std::string &value){
            myTree.add(key, value);

        }

        void add(const std::string &key, const double &value){
            myTree.add(key, value);

        }

        void add(const std::string &key, const int &value){
            myTree.add(key, value);

        }

        void add(const std::string &key, const bool &value){
            myTree.add(key, value);

        }

        void put(const std::string &key, const std::string &value){
            myTree.put(key, value);

        }

        void put(const std::string &key, const double &value){
            myTree.put(key, value);

        }

        void put(const std::string &key, const int &value){
            myTree.put(key, value);

        }

        void put(const std::string &key, const bool &value){
            myTree.put(key, value);

        }

        void write(const std::string &filename){
            std::ofstream out;
            out.open(filename, std::ofstream::out);
            boost::property_tree::write_info(out, myTree);
            out.close();

        }
            


};


BOOST_PYTHON_MODULE(propertytree){
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

}

