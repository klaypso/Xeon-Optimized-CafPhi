// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/caff