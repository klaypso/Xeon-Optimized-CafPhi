#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class HDF5DataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HDF5DataLayerTest()
      : filename(NULL),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_label2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    blob_top_vec_.push_back(blob_top_label2_);

    // Check out generate_sample_data.py in the same directory.
    filename = new string(
    CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data_list.txt" CMAKE_EXT);
    LOG(INFO)<< "Using sample HDF5 data file " << filename;
  }

  virtual ~HDF5DataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_top_label2_;
    delete filename;
  }

  string* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_top_label2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HDF5DataLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5DataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  // Create LayerParameter with the known parameters.
  // The data file we are reading has 10 rows and 8 columns,
  // with values from 0 to 10*8 reshaped in row-major order.