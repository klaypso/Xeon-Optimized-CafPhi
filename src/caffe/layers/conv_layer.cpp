#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


#include <pmmintrin.h>
#include <immintrin.h>
#include <cilk/cilk.h>
#include <cilk/reducer.h>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO) << "XEON conv_layer.cpp: Forward_cpu";
#endif
  const Dtype* weight = this->blobs_[0]->cpu_data();
#ifdef XEON_PHI
  for(int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    cilk_for(int n = 0; n < this->num_; ++n) {
      if(this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        /* Forward convolution */
        this->forward_convolution(bottom_data + bottom[i]->offset(n),
                                  weight,
                                  top_data + top[i]->offset(n),
                                  bias);
      }
      else{
        this->forward_convolution(bottom_data + bottom[i]->offset(n),
                                  weight,
                                  top_d