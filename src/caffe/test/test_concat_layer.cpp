#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ConcatLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConcatLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<ConstantFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0_);
    filler_param.set_value(2.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1_);
    filler_param.set_value(3.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConcatLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_bottom_2_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConcatLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConcatLayerTest, TestSetupNum) {
  typedef type