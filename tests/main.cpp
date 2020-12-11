//cfg parser
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

//logging
#include <boost/log/trivial.hpp>
#include <boost/log/common.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include "DMChannel.h"
#include "SpeckleToDM.h"
#include "SpeckleKalman.h"
#include "ImageGrabber.h"
#include "loopFunctions.h"
#include "dmTools.h"
#include <opencv2/opencv.hpp>


void addLogfile(const std::string &logfile, bool useConsoleLog){
    boost::log::core::get()->remove_all_sinks();
    boost::log::add_common_attributes();
    boost::log::add_file_log(
            boost::log::keywords::file_name = logfile,
            boost::log::keywords::format = 
                (boost::log::expressions::stream << "[" <<
                    boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
                    << "][" << boost::log::trivial::severity << "]: " << boost::log::expressions::smessage)
            
            );

    if(useConsoleLog)
        boost::log::add_console_log(std::cout,  
                boost::log::keywords::format = 
                    (boost::log::expressions::stream << "[" <<
                        boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
                        << "][" << boost::log::trivial::severity << "]: " << boost::log::expressions::smessage)
                );

}

int main()
{ 
    boost::property_tree::ptree cfgParams;
    read_info("speckNullConfig20200524.info", cfgParams);
    addLogfile("20200626.log", false);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

    loopfunctions::runLoop(500, cfgParams, true);
    //DMChannel chan("dm04disp00");
    //chan.save("dm04disp00");
    


}

