---
title: Siamese Network Tutorial
description: Train and test a siamese network on MNIST data.
category: example
include_in_docs: true
layout: default
priority: 100
---

# Siamese Network Training with Caffe
This example shows how you can use weight sharing and a contrastive loss
function to learn a model using a siamese network in Caffe.

We will assume that you have caffe successfully compiled. If not, please refer
to the [Installation page](../../installation.html). This example builds on the
[MNIST tutorial](mnist.html) so it would be a good idea to read that before
continuing.

*The guide specifies all paths and assumes all commands are executed from the
root caffe directory*

## Prepare Datasets

You will first need to download and convert the data from the MNIST
website. To do this, simply run the following commands:

    ./data/mnist/get_mnist.sh
    ./examples/siamese/create_mnist_siamese.sh

After running the script there should be two datasets,
`./examples/siamese/mnist_siamese_train_leveldb`, and
`./examples/siamese/mnist_siamese_test_leveldb`.

## The Model
First, we will define the model that we want to train using the siamese network.
We will use the convolutional net defined in
`./examples/siamese/mnist_siamese.prototxt`. This model is almost
exactly the same as the [LeNet model](mnist.html), the only difference is that
we have replaced the top layers that produced probabilities over the 10 digit
classes with a linear "feature" layer that produces a 2 dimensional vector.

    layers {
      name: "feat"
      type: INNER_PRODUCT
      bottom: "ip2"
      top: "feat"
      blobs_lr: 1
      blobs_lr: 2
      inner_product_param {
        num_output: 2
      }
    }

## Define the Siamese Network

In this section we will define the siamese network used for training. The
resulting network is defined in
`./examples/siamese/mnist_siamese_train_test.prototxt`.

### Reading in the Pair Data

We start with a data layer that reads from the LevelDB database we created
earlier. Each entry in this database contains the image data for a pair of
images (`pair_data`) and a binary label saying if they belong to the same class
or different classes (`sim`).

    layers {
      name: "pair_data"
      type: DATA
      top: "pair_data"
      top: "sim"
      data_param {
        source: "examples/siamese/mnist-siamese-train-leveldb"
        scale: 0.00390625
        batch_size: 64
      }
      include: { phase: TRAIN }
    }

In order to pack a pair of images into the same blob in the database we pack one
image per channel. We want to be able to work with these two images separately,
so we add a slice layer after the data layer. This takes the `pair_data` and
slices it along the channel dimension so that we have a single image in `data`
and its paired image in `data_p.`

    layers {
        name: "slice_pair"
        type: SLICE
        bottom: "pair_data"
        top: "data"
        top: "data_p"
        slice_param {
            slice_dim: 1
            slice_point: 1
        }
    }

### Building the First Side of the Siamese Net

Now we can specify the first side of the siamese net. This side operates 