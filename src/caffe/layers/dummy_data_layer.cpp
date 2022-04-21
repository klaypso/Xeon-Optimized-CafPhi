#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DummyDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_data_filler = param.data_filler_size();
  CHECK(num_data_filler == 0 || num_data_filler == 1 ||
        num_data_filler == num_top)
      << "Number of data fillers must be 0, 1 or equal to the number of tops: "
      << num_top << "; you specified " << num_data_filler << " data fillers.";

  const bool legacy_dims = param.num_size() || param.channels_size() ||
                           param.height_size() || param.width_si