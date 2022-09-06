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
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    shape { "
        "      dim: 5 "
        "      dim: 2 "
        "      dim: 3 "
        "      dim: 4 "
        "    } "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    shape { "
        "      dim: 5 "
        "    } "
        "    data_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    if (accuracy_layer) {
      proto +=
          "layer { "
          "  name: 'loss' "
          "  type: 'Accuracy' "
          "  bottom: 'innerproduct' "
          "  bottom: 'label' "
          "  top: 'accuracy' "
          "} ";
    }
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTinyNetEuclidean(const bool force_backward = false) {
    string proto =
        "name: 'TinyTestEuclidLossNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTrickyNet(Dtype* loss_weight = NULL) {
    ostringstream loss_weight_stream;
    if (loss_weight) {
      loss_weight_stream << "  loss_weight: " << *loss_weight << " ";
    }
    const string& proto =
        "name: 'TrickyTestNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'transformed_data' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'label' "
        "  top: 'transformed_label' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' " +
        loss_weight_stream.str() +
        "  bottom: 'transformed_data' "
        "  bottom: 'transformed_label' "
        "} ";
    InitNetFromProtoString(proto);
  }

  // loss_weight is the loss weight for the 'EuclideanLoss' layer output.
  // midnet_loss_weight is the loss weight for the first 'InnerProduct' layer
  // output.  Should both default to 0.0 if unspecified (i.e., if NULL is
  // passed to this function).
  virtual void InitUnsharedWeightsNet(const Dtype* loss_weight = NULL,
      const Dtype* midnet_loss_weight = NULL,
      const bool force_backward = false, const bool bias_term = false,
      const Dtype blobs_lr_w1 = 1, const Dtype blobs_lr_b1 = 2,
      const Dtype blobs_lr_w2 = 1, const Dtype blobs_lr_b2 = 2) {
    ostringstream proto;
    proto << "name: 'UnsharedWeightsNetwork' ";
    if (force_backward) {
      proto << "force_backward: true ";
    }
    proto <<
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: " << bias_term <<
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { "
        "    name: 'unsharedweights1' "
        "    lr_mult: " << blobs_lr_w1 <<
        "  } ";
    if (bias_term) {
      proto << "  param { lr_mult: " << blobs_lr_b1 << " } ";
    }
    proto <<
        "  bottom: 'data' "
        "  top: 'innerproduct1' ";
    if (midnet_loss_weight) {
      proto << "  loss_weight: "