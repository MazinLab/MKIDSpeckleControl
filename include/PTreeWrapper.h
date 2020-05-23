#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

class PTreeWrapper{
    boost::property_tree::ptree myTree;
    
    public:
        void read_info(const std::string &filename){
            boost::property_tree::info_parser::read_info(filename, myTree);

        }

        std::string get(const std::string &key){
            return myTree.get<std::string>(key);

        }

        boost::property_tree::ptree getCXXObject(){
            return myTree;

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
