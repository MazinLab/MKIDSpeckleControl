#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/python.hpp>

using namespace boost::python;

//int (boost::property_tree::ptree::*get)(const boost::property_tree::ptree::path_type) = &boost::property_tree::ptree::get<int>;
//std::string (boost::property_tree::ptree::*get) (const boost::property_tree::ptree::path_type) = &boost::property_tree::ptree::get<std::string>;
//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads_int, boost::property_tree::ptree::get<int>, 1, 3);
//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads_float, boost::property_tree::ptree::get<float>, 1, 3);
//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads_double, boost::property_tree::ptree::get<double>, 1, 3);
//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads_string, boost::property_tree::ptree::get<std::string>, 1, 3);

//BOOST_PYTHON_MODULE(propertytree){
//        class_<boost::property_tree::ptree>("PTree")
//            .def("get", &boost::property_tree::ptree::get<int>, get_overloads_int());
//       
//
//}

//BOOST_PYTHON_FUNCTION_OVERLOADS(read_info_overloads, boost::property_tree::info_parser::read_info, 2, 2)

void read_info_wrapper(const std::string &filename, boost::property_tree::ptree &tree){
    boost::property_tree::info_parser::read_info(filename, tree);}

std::string ptree_get_string(const boost::property_tree::ptree &tree, std::string &key){
    return tree.get<std::string>(key);}

class PTreeWrapper{
    boost::property_tree::ptree myTree;
    
    void read_info(std::string &filename){
        boost::property_tree::info_parser::read_info(filename, myTree);

    }

    std::string get(std::string &key){
        return myTree.get<std::string>(key);

    }

};



BOOST_PYTHON_MODULE(propertytree){
        class_<boost::property_tree::ptree>("PTree")
            //.def("get", static_cast<std::string (boost::property_tree::ptree::*)(const boost::property_tree::ptree::path_type&) const>(&boost::property_tree::ptree::get<std::string>));
            .def("get", static_cast<std::string (boost::property_tree::ptree::*)(const boost::property_tree::ptree::path_type&) const>(&boost::property_tree::ptree::get<std::string>));
            //.def("get", ((std::string (boost::property_tree::ptree*)(const boost::property_tree::ptree::path_type&) const)(&boost::property_tree::ptree::get<std::string>)));
            //.def("get", static_cast<int (boost::property_tree::ptree::*)(const boost::property_tree::ptree::path_type&) const>(&boost::property_tree::ptree::get<int>))
            //.def("get", static_cast<double (boost::property_tree::ptree::*)(const boost::property_tree::ptree::path_type&) const>(&boost::property_tree::ptree::get<double>));

        //def("read_info", static_cast<void (*)(const std::string &, boost::property_tree::ptree &, const std::locale &loc)>(&boost::property_tree::info_parser::read_info));
        def("read_info", &read_info_wrapper);
        def("get", &ptree_get_string);

}

