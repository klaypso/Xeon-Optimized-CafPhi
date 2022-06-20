#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
WindowDataLayer<Dtype>::~WindowDataLayer<D