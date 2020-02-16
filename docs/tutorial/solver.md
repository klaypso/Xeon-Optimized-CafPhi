---
title: Solver / Model Optimization
---
# Solver

The solver orchestrates model optimization by coordinating the network's forward inference and backward gradients to form parameter updates that attempt to improve the loss.
The responsibilities of learning are divided between the Solver for overseeing the optimization and generating parameter updates and the Net for yielding loss and gradients.

The Caffe solvers are Stochastic Gradient Descent (SGD), Adaptive Gradient (ADAGRAD), and Nesterov's Accelerated Gradient (NESTEROV).

The solver

1. scaffolds the optimization bookkeeping and creates the training network for learning and test network(s) for evaluation.
2. iteratively optimizes by calling forward / backward and updating parameters
3. (periodically) evaluates the test networks
4. snapshots the model and solver state throughout the optimization

where each iteration

1. calls network forward to compute the output and loss
2. calls network backward to compute the gradients
3. incorporates the gradients into parameter updates according to the solver method
4. updates the solver state according to learning rate, history, and method

to take the weights all the way from initialization to learned model.

Like Caffe models, Caffe solvers run in CPU / GPU modes.

## Methods

The solver methods address the general optimization problem of loss minimization.
For dataset $$D$$, the optimization objective is the average loss over all $$|D|$$ data instances throughout the dataset

$$L(W) = \frac{1}{|D|} \sum_i^{|D|} f_W\left(X^{(i)}\right) + \lambda r(W)$$

where $$f_W\left(X^{(i)}\right)$$ is the loss on data instance $$X^{(i)}$$ and $$r(W)$$ is a regularization term with weight $$\lambda$$.
$$|D|$$ can be very large, so in practice, in each solver iteration we use a stochastic approximation of this objective, drawing a mini-batch of $$N << |D|$$ instances:

$$L(W) \approx \frac{1}{N} \sum_i^N f_W\left(X^{(i)}\right) + \lambda r(W)$$

The model computes $$f_W$$ in the forward pass and the gradient $$\nabla f_W$$ in the backward pass.

The parameter update $$\Delta W$$ is formed by the solver from the error gradient $$\nabla f_W$$, the regularization gradient $$\nabla r(W)$$, and other particulars to each method.

### SGD

**Stochastic gradient descent** (`solver_type: SGD`) updates the weights $$ W $$ by a linear combination of the negative gradient $$ \nabla L(W) $$ and the previous weight update $$ V_t $$.
The **learning rate** $$ \alpha $$ is the weight of the negative gradient.
The **momentum** $$ \mu $$ is the weight of the previous update.

Formally, we have the following formulas to compute the update value $$ V_{t+1} $$ and the updated weights $$ W_{t+1} $$ at iteration $$ t+1 $$, given the previous weight update $$ V_t $$ and current weights $$ W_t $$:

$$
V_{t+1} = \mu V_t - \alpha \nabla L(W_t)
$$

$$
W_{t+1} = W_t + V_{t+1}
$$

The learning "hyperparameters" ($$\alpha$$ and $$\mu$$) might require a bit of tuning for best results.
If you're not sure where to start, take a look at the "Rules of thumb" below, and for further information you might refer to Leon Bottou's [Stochastic Gradient Descent Tricks](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) [1].

[1] L. Bottou.
    [Stochastic Gradient Descent Tricks](http://research.microsoft.com/pubs/192769/tricks-2012.pdf).
    *Neural Networks: Tricks of the Trade*: Springer, 2012.

#### Rules of thumb for setting the learning rate $$ \alpha $$ and momentum $$ \mu $$

A good strategy for deep learning with SGD is to initialize the learning rate $$ \alpha $$ to a value around $$ \alpha \approx 0.01 = 10^{-2} $$, and dropping it by a constant factor (e.g., 10) throughout training when the loss begins to reach an apparent "plateau", repeating this several times.
Generally, you probably want to use a momentum $$ \mu = 0.9 $$ or similar value.
By smoothing the weight updates across iterations, momentum tends to make deep learning with SGD both stabler and faster.

This was the strategy used by Krizhevsky et al. [1] in their famously winning CNN entry to the ILSVRC-2012 competition, and Caffe makes this strategy easy to implement in a `SolverParameter`, as in our reproduction of [1] at `./examples/imagenet/alexnet_solver.prototxt`.

To use a learning rate policy like this, you can put the following lines somewhere in your solver prototxt file:

    base_lr: 0.01     # begin training at a learning rate of 0.01 = 1e-2

    lr_policy: "step" # learning rate policy: drop the learning rate in "steps"
                      # by a factor of gamma every stepsize iterations

    gamma: 0.1        # drop the learning rate by a factor of 10
                      # (i.e., multiply it by a factor of gamma = 0.1)

    stepsize: 100000  # drop the learning rate every 100K iterations

    max_iter: 350000  # train for 350K iterations total

    momentum: 0.9

Under the above settings, we'll always use `momentum` $$ \mu = 0.9 $$.
We'll begin training at a `base_lr` of $$ \alpha = 0.01 = 10^{-2} $$ for the first 100,000 iterations, then multiply the learning rate by `gamma` ($$ \gamma $$) and train at $$ \alpha' = \alpha \gamma = (0.01) (0.1) = 0.001 = 10^{-3} $$ for i