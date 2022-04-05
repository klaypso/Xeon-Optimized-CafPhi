
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <pmmintrin.h>
#include <immintrin.h>


namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO) << "XEON group:" << group_;
#endif
  CHECK_EQ(group_, 1)<<"ROHITH: Bad assumption";
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
#ifdef XEON_PHI_ESSENTIAL_DEBUG
    LOG(INFO)<<" \t\tid:"<< top_id <<" num:"<< num_<<" numoutput:"
	     <<num_output_<<" height:"<<height_out_<<" width_out:"
	     << width_out_;
#endif
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
#ifdef XEON_PHI_ESSENTIAL_DEBUG
    LOG(INFO)<<"---> Reverse: kdim:"<< kernel_dim_ <<" height:"<< height_<<
	       " width:"<< width_;
#endif
    col_buffer_.Reshape(num_, kernel_dim_, height_, width_);
  } else {
#ifdef XEON_PHI_ESSENTIAL_DEBUG
    LOG(INFO)<<"---> kdim:"<< kernel_dim_ <<" height:"<< height_out_<<
	       " width:"<< width_out_;
#endif
    col_buffer_.Reshape(num_, kernel_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {

#ifdef XEON_PHI_ESSENTIAL_DEBUG
    LOG(INFO)<<"---->M:"<<conv_out_channels_/group_<<" N:"<<
      conv_out_spatial_dim_<<" K:"<< kernel_dim_ / group_<<" alpha:1"
      <<" beta:0";
#endif

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
















template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, int n, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data() + col_offset_ * n);
    }
    col_buff = col_buffer_.cpu_data() + col_offset_ * n;
  }
  for (int g = 0; g < group_; ++g) {

#ifdef XEON_PHI_ESSENTIAL_DEBUG
    LOG(INFO)<<"---->M:"<<conv_out_channels_/group_<<" N:"<<
      conv_out_spatial_dim_<<" K:"<< kernel_dim_ / group_<<" alpha:1"
      <<" beta:0";
#endif

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias, int n) {

#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO)<<"\t\t\t\t---->M:"<<num_output_<<" N:"<<(height_out_ * width_out_)<<
  " K:1 alpha:1 beta:1";
#endif

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input, int n) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data() + col_offset_ * n;
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights, int n) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data() + col_offset_ * n);
    col_buff = col_buffer_.cpu_data() + col_offset_ * n;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input, int n) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}



























template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {

#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO)<<"\t\t\t\t---->M:"<<num_output_<<" N:"<<(height_out_ * width_out_)<<
	" K:1 alpha:1 beta:1";
#endif

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}


#ifdef XEON_PHI

/* Perform convolution using for loops */
void BaseConvolutionLayer<float>::forward_convolution(const float* input,
    const float* weight, float* output, const float *bias) {

  float b[8];
  float *a = b;

#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO)<<"inChan:"<<conv_in_channels_<<" outChan:"<<
	     conv_out_channels_<<" height_out:"<<height_out_<<
	     " width_out:"<<width_out_<<" kernel_h:"<<kernel_h_<<
	     " kernel_w:"<<kernel_w_;
#endif

  if(!bias){
    memset(output, 0, conv_out_channels_ * height_out_ * width_out_ * sizeof(float));
  }
  else{
    for(int outChan = 0; outChan < conv_out_channels_; ++outChan) {
      /* Loop over output image height */
      for(int h = 0; h < height_out_; ++h) {
        /* Loop over output image width */
        for(int w = 0; w < width_out_; ++w) {
          output[outChan * height_out_ * width_out_ + h * width_out_ + w] = bias[outChan];
        }
      }
    }
  }


/* TODO: what about group? */
  //for (int g = 0; g < group_; ++g) {
  /* Loop for number of input channels */
  for(int inChan = 0; inChan < conv_in_channels_; ++inChan) {
    /* Loop over all output channels */
    for(int outChan = 0; outChan < conv_out_channels_; ++outChan) {
      /* Loop over output image height */
      for(int h = 0; h < height_out_; ++h) {
        /* Loop over output image width */
        for(int w = 0; w < 8*(width_out_/8); w+=8) {

          /* Loop over kernel image height */
          for(int i = 0; i < kernel_h_; ++i) {
            /* Loop over kernel image width */
            for(int j = 0; j < kernel_w_; ++j) {

              __m256 num1 = _mm256_setr_ps (input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+0) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+1) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+2) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+3) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+4) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+5) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+6) + j],
                                         input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+7) + j]);
              __m256 num2 = _mm256_set1_ps (weight[outChan * conv_in_channels_ * kernel_h_ * kernel_w_ +
                  inChan * kernel_h_ * kernel_w_ + i * kernel_w_ + j]);

              // __m256 num3 = _mm256_setzero_ps();

              __m256 num3 = _mm256_loadu_ps(output + outChan * height_out_ * width_out_ + h * width_out_ + w);

              // num3 = _mm256_fmadd_ps(num1, num2, num3);

              __m256 num4 = _mm256_mul_ps(num1, num2);

              num4 = _mm256_add_ps(num4, num3);

              _mm256_storeu_ps (output + outChan * height_out_ * width_out_ + h * width_out_ + w, num4);



              // LOG(INFO)<<"avx shit 2";
              // for(int d = 0; d < 8; ++d)
                // *(__m256 *)output[outChan * height_out_ * width_out_ + h * width_out_ + w] = num3;

            }
          }
        }

        for(int w = 8*(width_out_/8); w < width_out_; w++) {
          /* Loop over kernel image height */
          for(int i = 0; i < kernel_h_; ++i) {
            /* Loop over kernel image width */
            for(int j = 0; j < kernel_w_; ++j) {

              output[outChan * height_out_ * width_out_ + h * width_out_ + w] +=
                  input[inChan * conv_in_height_ * conv_in_width_ +
                             (h+i) * conv_in_width_ + (w+j)] *
                                 weight[outChan * conv_in_channels_ * kernel_h_ * kernel_w_ +
                  inChan * kernel_h_ * kernel_w_ +
                  i * kernel_w_ + j];
            }
          }
        }
      }
    }
  }
  //}






























//   memset(output, 0, conv_out_channels_ * height_out_ * width_out_ * sizeof(Dtype));

//   /* TODO: what about group? */
//   //for (int g = 0; g < group_; ++g) {
//   /* Loop for number of input channels */
//   for(int inChan = 0; inChan < conv_in_channels_; ++inChan) {
//     /* Loop over all output channels */
//     for(int outChan = 0; outChan < conv_out_channels_; ++outChan) {
//       /* Loop over output image height */
//       for(int h = 0; h < height_out_; ++h) {
// 	/* Loop over output image width */
// 	for(int w = 0; w < width_out_; ++w) {
// 	   //Loop over kernel image height
// 	  for(int i = 0; i < kernel_h_; ++i) {
// 	    /* Loop over kernel image width */
// 	    for(int j = 0; j < kernel_w_; ++j) {

// 	      output[outChan * height_out_ * width_out_ + h * width_out_ + w] +=
// 	          input[inChan * conv_in_height_ * conv_in_width_ +
// 		                   (h+i) * conv_in_width_ + (w+j)] *
//     	                     weight[outChan * conv_in_channels_ * kernel_h_ * kernel_w_ +
// 				    inChan * kernel_h_ * kernel_w_ +
// 				    i * kernel_w_ + j];
// 	    }
// 	  }
// 	}
//       }
//     }
//   }
//   //}
}



/*
 * Performing the equivalent of:
 * output = alpha * bias * input + beta * output
 * where alpha = beta = 1
 */
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_bias(Dtype* output,
    const Dtype* bias) {
  const Dtype* input = bias_multiplier_.cpu_data();

#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO)<<" outChan:"<<conv_out_channels_<<" height_out:"<<
	     height_out_<<" width_out:"<<width_out_;
#endif

  /* Loop over all output channels */
  for(int outChan = 0; outChan < num_output_; ++outChan) {
    /* Loop over output image height */
    for(int h = 0; h < height_out_; ++h) {
      /* Loop over output image width */
      for(int w = 0; w < width_out_; ++w) {
	output[outChan * height_out_ * width_out_ + width_out_ * h + w] +=
	    bias[outChan] * input[h * width_out_ + w];
      }
    }
  }
}

#endif /* XEON_PHI */


#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe