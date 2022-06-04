#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // do mean and variance normalization
    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX

    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward_cpu(const vector