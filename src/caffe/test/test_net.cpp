#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NetTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NetTest() : seed_(1701) {}

  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void CopyNetBlobs(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* blobs_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_blobs = net_->blobs();
    blobs_copy->clear();
    blobs_copy->resize(net_blobs.size());
    const bool kReshape = true;
    for (int i = 0; i < net_blobs.size(); ++i) {
      (*blobs_copy)[i].reset(new Blob<Dtype>());
      (*blobs_copy)[i]->CopyFrom(*net_blobs[i], copy_diff, kReshape);
    }
  }

  virtual void CopyNetParams(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* params_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
    params_copy->clear();
    params_copy->resize(net_params.size());
    const bool kReshape = true;
    for (int i = 0; i < net_params.size(); ++i) {
      (*params_copy)[i].reset(new Blob<Dtype>());
      (*params_copy)[i]->CopyFrom(*net_params[i], copy_diff, kReshape);
    }
  }

  virtual void InitTinyNet(const bool force_backward = false,
                           const bool accuracy_layer = false) {
    string proto =
        "name: 'TinyTestNetwork' "
      