#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename TypeParam>
class HDF5OutputLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HDF5OutputLayerTest()
      : input_file_name_(
        CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data.h5"),
        blob_data_(new Blob<Dtype>()),
        blob_label_(new Blob<Dtype>()),
        num_(5),
        channels_(8),
        height_(5),
        width_(5) {
    MakeTempFilename(&output_file_name_);
  }

  virtual ~HDF5OutputLayerTest() {
    delete blob_data_;
    delete blob_label_;
  }

  void CheckBlobEqual(const Blob<Dtype>& b1, const Blob<Dtype>& b2);

  string output_file_name_;
  string input_file_name_;
  Blob<Dtype>* const blob_data_;
  Blob<Dtype>* const blob_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int num_;
  int channels_;
  int height_;
  int width_;
};

template<typename TypeParam>
void HDF5OutputLayerTest<TypeParam>::CheckBlobEqual(const Blob<Dtype>& b1,
                                                    const Blob<Dtype>& b2) {
  EXPECT_EQ(b1.num(), b2.num());
  EXPECT_EQ(b1.channels(), b2.channels());
  EXPECT_EQ(b1.height(), b2.height());
  EXPECT_EQ(b1.width(), b2.width());
  for (int n = 0; n < b1.num(); ++n) {
    for (int c = 0; c < b1.channels(); ++c) {
      for (int h = 0; h < b1.height(); ++h) {
        for (int w = 0; w < b1.width(); ++w) {
          EXPECT_EQ(b1.data_at(n, c, h, w), b2.data_at(n, c, h, w));
        }
      }
 