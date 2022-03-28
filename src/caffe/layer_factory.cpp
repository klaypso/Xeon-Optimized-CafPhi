#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#ifdef WITH_PYTHON_LAYER
#include "caffe/python_layer.hpp"
#endif

namespace caffe {

// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ConvolutionParameter_Engine_CUDNN;
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    PoolingParameter p_param = param.pooling_param();
    if (p_param.pad() || p_param.pad_h() || p_param.pad_w() ||
        param.top_size() > 1) {
      LOG(INFO) << "CUDNN does not support padding or multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    }