---
title: Blobs, Layers, and Nets
---
# Blobs, Layers, and Nets: anatomy of a Caffe model

Deep networks are compositional models that are naturally represented as a collection of inter-connected layers that work on chunks of data. Caffe defines a net layer-by-layer in its own model schema. The network defines the entire model bottom-to-top from input data to loss. As data and derivatives flow through the network in the [forward and backward p