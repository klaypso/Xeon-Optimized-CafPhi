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
  // First, compute the loss and gradients with no loss_weight specified.
  // In this case, the loss weight for the 'EuclideanLoss' layer should default
  // to 1.
  vector<Blob<Dtype>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  this->InitUnsharedWeightsNet(NULL, NULL, kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<Dtype> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<Dtype> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const Dtype kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const Dtype kErrorMargin = 1e-4;
  const int kNumLossWeights = 6;
  Dtype kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&kLossWeights[i], NULL, kForceBackward);
    const Dtype weighted_loss = this->net_->ForwardBackward(bottom);
    const Dtype error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const vector<shared_ptr<Blob<Dtype> > >& weighted_blobs =
        this->net_->blobs();
    ASSERT_EQ(blob_grads.size(), weighted_blobs.size());
    for (int j = 0; j < blob_grads.size(); ++j) {
      ASSERT_EQ(blob_grads[j]->count(), weighted_blobs[j]->count());
      for (int k = 0; k < blob_grads[j]->count(); ++k) {
        EXPECT_NEAR(blob_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_blobs[j]->cpu_diff()[k], error_margin);
      }
    }
    const vector<shared_ptr<Blob<Dtype> > >& weighted_params =
        this->net_->params();
    ASSERT_EQ(param_grads.size(), weighted_params.size());
    for (int j = 0; j < param_grads.size(); ++j) {
      ASSERT_EQ(param_grads[j]->count(), weighted_params[j]->count());
      for (int k = 0; k < param_grads[j]->count(); ++k) {
        EXPECT_NEAR(param_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_params[j]->cpu_diff()[k], error_margin);
      }
    }
  }
}

TYPED_TEST(NetTest, TestLossWeightMidNet) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  Dtype loss_weight = 0;
  Dtype midnet_loss_weight = 1;
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  const bool kReshape = true;
  Blob<Dtype> data_grad;
  data_grad.CopyFrom(*this->net_->blob_by_name("data"), kCopyDiff, kReshape);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const Dtype kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const Dtype kErrorMargin = 1e-4;
  const int kNumLossWeights = 6;
  Dtype kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&loss_weight, &kLossWeights[i],
                                 kForceBackward);
    const Dtype weighted_loss = this->net_->ForwardBackward(bottom);
    const Dtype error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const shared_ptr<Blob<Dtype> >& weighted_blob =
        this->net_->blob_by_name("data");
    ASSERT_EQ(data_grad.count(), weighted_blob->count());
    for (int j = 0; j < data_grad.count(); ++j) {
      EXPECT_NEAR(data_grad.cpu_diff()[j] * kLossWeights[i],
                  weighted_blob->cpu_diff()[j], error_margin);
    }
  }
}

TYPED_TEST(NetTest, TestComboLossWeight) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  Dtype loss_weight;
  Dtype midnet_loss_weight;
  const bool kForceBackward = true;
  const Dtype kErrorMargin = 1e-4;

  // Get the loss and gradients with 'EuclideanLoss' weight 1,
  // 'InnerProduct' weight 1.
  loss_weight = 1;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<Dtype> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<Dtype> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);

  loss_weight = 2;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_main_2 = this->net_->ForwardBackward(bottom);
  vector<shared_ptr<Blob<Dtype> > > blob_grads_loss_2;
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  vector<shared_ptr<Blob<Dtype> > > param_grads_loss_2;
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 3;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_main_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<Dtype> > >& blob_grads_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_loss_3.size());
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = this->net_->blob_names()[j];
    bool grad_should_change = true;
    if (blob_name == "innerproduct1_innerproduct1_0_split_0") {
      grad_should_change = false;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_3[j]->count());
    ASSERT_EQ(blob_grads_loss_2[j]->count(), blob_grads_loss_3[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const Dtype grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const Dtype grad_diff_3 = blob_grads_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const Dtype kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  loss_weight = 1;
  midnet_loss_weight = 2;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_midnet_2 = this->net_->ForwardBackward(bottom);
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 1;
  midnet_loss_weight = 3;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_midnet_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<Dtype> > >& blob_grads_midnet_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_midnet_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_midnet_loss_3.size());
  const vector<string>& blob_names = this->net_->blob_names();
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = blob_names[j];
    bool grad_should_change = false;
    if (blob_name == "innerproduct1" ||
        blob_name == "innerproduct1_innerproduct1_0_split_0" ||
        blob_name == "data_data_0_split_0" || blob_name == "data") {
      grad_should_change = true;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_midnet_loss_3[j]->count());
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_2[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const Dtype grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const Dtype grad_diff_3 = blob_grads_midnet_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const Dtype kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  const Dtype kMinLossDiffAbsValue = 1e-4;

  Dtype loss_diff_2 = loss_main_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  Dtype loss_diff_3 = loss_main_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);

  loss_diff_2 = loss_midnet_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  loss_diff_3 = loss_midnet_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);
}

TYPED_TEST(NetTest, TestBackwardWithAccuracyLayer) {
  typedef typename TypeParam::Dtype Dtype;
  const bool kForceBackward = false;
  const bool kAccuracyLayer = true;
  this->InitTinyNet(kForceBackward, kAccuracyLayer);
  EXPECT_TRUE(this->net_->has_blob("accuracy"));
  vector<Blob<Dtype>*> bottom;
  // Test that we can do Backward even though we have an 'Accuracy' layer.
  this->net_->ForwardBackward(bottom);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDataNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitUnsharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Dtype loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_GT(loss, 0);
}

TYPED_TEST(NetTest, TestSharedWeightsDataNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitSharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Dtype loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_FLOAT_EQ(loss, 0);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDiffNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitUnsharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Net<Dtype>* net = this->net_.get();
  net->Forward(bottom);
  net->Backward();
  Layer<Dtype>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<Dtype>* ip2_layer = net->layer_by_name("innerproduct2").get();
  const int count = ip1_layer->blobs()[0]->count();
  const Dtype* grad1 = ip1_layer->blobs()[0]->cpu_diff();
  const Dtype* grad2 = ip2_layer->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_GT(fabs(grad1[i]), 0);
    EXPECT_FLOAT_EQ(-1 * grad1[i], grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsDiffNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitSharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Net<Dtype>* net = this->net_.get();
  Dtype loss;
  net->Forward(bottom, &loss);
  net->Backward();
  EXPECT_FLOAT_EQ(loss, 0);
  Layer<Dtype>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<Dtype>* ip2_layer = net->layer_by_name("innerproduct2").get();
  c