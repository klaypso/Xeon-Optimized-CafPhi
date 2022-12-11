#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
#if XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO)<<"\t\t\tim2col:channels="<< channels <<" h="<< height;
  LOG(INFO)<<"\t\t\t       w="<< width <<" kernel_h="<< kernel_h;
  LOG(INFO)<<"\t\t\t       kernel_w="<< kernel_w <<" pad_h="<< pad_h;
  LOG(INFO)<<"\t\t\t       pad_w="<< pad_w <<" stride_h="<< stride_h;
  LOG(INFO)<<"\t\t\t       stride_w="<< stride_w;
  LOG(INFO)<<"\t\t\t       h_col="<< height_col <<" w_col="<< width_col;
  LOG(INFO)<<"\t\t\t       channels_col="<< channels_col << "\n";
#endif
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
    