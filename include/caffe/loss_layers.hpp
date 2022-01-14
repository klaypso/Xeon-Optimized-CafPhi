#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Accuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy: @f$
   *        \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
   *      @f$, where @f$
   *      \delta\{\mathrm{condition}\} = \left\{
   *         \begin{array}{lr}
   *            1 & \mbox{if condition} \\
   *            0 & \mbox{otherwise}
   *         \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int label_axis_, outer_num_, inner_num_;

  int top_k_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
};

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

/**
 * @brief Computes the contrastive loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d +
 *              \left(1-y\right) \max \left(margin-d, 0\right)
 *          @f$ where @f$
 *          d = \left| \left| a_n - b_n \right| \right|_2^2 @f$. This can be
 *          used to train siamese networks.
 *
 * @param bottom input Blob vector (length 3)
 *   -# @f$ (N \times C \times 1 \times 1) @f$
 *      the features @f$ a \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times 1 \times 1) @f$
 *      the features @f$ b \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the binary similarity @f$ s \in [0, 1]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed contrastive loss: @f$ E =
 *          \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d +
 *          \left(1-y\right) \max \left(margin-d, 0\right)
 *          @f$ where @f$
 *          d = \left| \left| a_n - b_n \right| \right|_2^2 @f$.
 * This can be used to train siamese networks.
 */
template <typename Dtype>
class ContrastiveLossLayer : public LossLayer<Dtype> {
 public:
  explicit ContrastiveLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "ContrastiveLoss"; }
  /**
   * Unlike most loss layers, in the ContrastiveLossLayer we can backpropagate
   * to the first two inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Contrastive error gradient w.r.t. the inputs.
   *
   * Computes the gradients with respect to the two input vectors (bottom[0] and
   * bottom[1]), but not the similarity label (bottom[2]).
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times 1 \times 1) @f$
   *      the features @f$a@f$; Backward fills their diff with
   *      gradients 