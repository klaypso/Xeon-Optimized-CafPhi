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
      proto << "  loss_weight: " << *midnet_loss_weight << " ";
    }
    proto <<
        "} "
        "layer { "
        "  name: 'innerproduct2' "
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
        "    name: 'unsharedweights2' "
        "    lr_mult: " << blobs_lr_w2 <<
        "  } ";
    if (bias_term) {
      proto << "  param { lr_mult: " << blobs_lr_b2 << " } ";
    }
    proto <<
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' ";
    if (loss_weight) {
      proto << "  loss_weight: " << *loss_weight << " ";
    }
    proto <<
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto.str());
  }

  virtual void InitSharedWeightsNet() {
    const string& proto =
        "name: 'SharedWeightsNetwork' "
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
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataUnsharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataUnsharedWeightsNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'unsharedweights1' } "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'unsharedweights2' } "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataSharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataSharedWeightsNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitReshapableNet() {
    const string& proto =
        "name: 'ReshapableNetwork' "
        "input: 'data' "
        "input_dim: 1 "
        "input_dim: 3 "
        "input_dim: 100 "
        "input_dim: 100 "
        "layer { "
        "  name: 'conv1' "
        "  type: 'Convolution' "
        "  bottom: 'data' "
        "  top: 'conv1' "
        "  convolution_param { "
        "    num_output: 5 "
        "    kernel_size: 3 "
        "    stride: 2 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0.2 "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'relu1' "
        "  type: 'ReLU' "
        "  bottom: 'conv1' "
        "  top: 'conv1' "
        "} "
        "layer { "
        "  name: 'pool1' "
        "  type: 'Pooling' "
        "  bottom: 'conv1' "
        "  top: 'pool1' "
        "  pooling_param { "
        "    pool: MAX "
        "    kernel_size: 2 "
        "    stride: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'norm1' "
        "  type: 'LRN' "
        "  bottom: 'pool1' "
        "  top: 'norm1' "
        "  lrn_param { "
        "    local_size: 3 "
        "  } "
        "} "
        "layer { "
        "  name: 'softmax' "
        "  type: 'Softmax' "
        "  bottom: 'norm1' "
        "  top: 'softmax' "
        "} ";
    InitNetFromProtoString(proto);
  }

  int seed_;
  shared_ptr<Net<Dtype> > net_;
};

TYPED_TEST_CASE(NetTest, TestDtypesAndDevices);

TYPED_TEST(NetTest, TestHasBlob) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
  EXPECT_TRUE(this->net_->has_blob("top_loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->blob_by_name("data"), this->net_->blobs()[0]);
  EXPECT_EQ(this->net_->blob_by_name("label"), this->net_->blobs()[1]);
  EXPECT_EQ(this->net_->blob_by_name("innerproduct"), this->net_->blobs()[2]);
  EXPECT_FALSE(this->net_->blob_by_name("loss"));
  EXPECT_EQ(this->net_->blob_by_name("top_loss"), this->net_->blobs()[3]);
}

TYPED_TEST(NetTest, TestHasLayer) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_layer("data"));
  EXPECT_TRUE(this->net_->has_layer("innerproduct"));
  EXPECT_TRUE(this->net_->has_layer("loss"));
  EXPECT_FALSE(this->net_->has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->layer_by_name("data"), this->net_->layers()[0]);
  EXPECT_EQ(this->net_->layer_by_name("innerproduct"), this->net_->layers()[1]);
  EXPECT_EQ(this->net_->layer_by_name("loss"), this->net_->layers()[2]);
  EXPECT_FALSE(this->net_->layer_by_name("label"));
}

TYPED_TEST(NetTest, TestBottomNeedBackward) {
  this->InitTinyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardForce) {
  const bool force_backward = true;
  this->InitTinyNet(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardEuclideanForce) {
  const bool force_backward = true;
  this->InitTinyNetEuclidean(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(true, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardTricky) {
  this->InitTrickyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(4, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(1, bottom_need_backward[2].size());
  EXPECT_EQ(false, bottom_need_backward[2][0]);
  EXPECT_EQ(2, bottom_need_backward[3].size());
  EXPECT_EQ(true, bottom_need_backward[3][0]);
  // The label input to the SoftmaxLossLayer should say it "needs backward"
  // since it has weights under it, even though we expect this to cause a crash
  // at training/test time.
  EXPECT_EQ(true, bottom_need_backward[3][1]);
}

TYPED_TEST(NetTest, TestLossWeight) {
  typedef typename TypeParam::Dtype Dtype;
  // First, compute the loss and gradients with no 