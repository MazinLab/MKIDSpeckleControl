#include "DMChannel.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

int main()
{ 
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    DMChannel chan("dm03disp03");
    chan.getBufferPtr<uint8_t>();

}
