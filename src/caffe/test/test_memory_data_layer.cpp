#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MemoryDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MemoryDataLayerTest()
    : data_(new Blob<Dtype>()),
      labels_(new Blob<Dtype>()),
      data_blob_(new Blob<Dtype>()),
      label_blob_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    batch_size_ = 8;
    batches_ = 12;
    channels_ = 4;
    height_ = 7;
    width_ = 11;
    blob_top_vec_.push_back(data_blob_);
    blob_top_vec_.push_back(label_blob_);
    // pick random input data
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    data_->Reshape(batches_ * batch_size_, channels_, height_, width_);
    labels_->Reshape(batches_ * batch_size_, 1, 1, 1);
    filler.Fill(this->data_);
    filler.Fill(this->labels_);
  }

  virtual ~MemoryDataLayerTest() {
    delete data_blob_;
    delete label_blob_;
    delete data_;
    delete labels_;
  }
  int batch_size_;
  int batches_;
  int channels_;
  int height_;
  int width_;
  // we don't really need blobs for the input data, but it makes it
  //  easier to call Filler
  Blob<Dtype>* const data_;
  Blob<Dtype>* const labels_;
  // blobs for the top of MemoryDataLayer
  Blob<Dtype>* const data_blob_;
  Blob<Dtype>* const label_blob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MemoryDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MemoryDataLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  MemoryDataParameter* md_param = layer_param.mutable_memory_data_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_channels(this->channels_);
  md_param->set_height(this->height_);
  md_param->set_width(this->width_);
  shared_ptr<Layer<Dtype> > layer(
      new MemoryDataLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bott