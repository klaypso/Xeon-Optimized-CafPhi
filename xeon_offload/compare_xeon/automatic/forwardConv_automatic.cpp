/* Implicit offload of computation on Xeon-Phi 
 * Size chosen based on:
 * VSM size exceeds the limitation (17179869184) now!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>
#include <unistd.h>
#include <mkl.h>

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include "CycleTimer.h"


#define MIN_VAL         (-1)
#define MAX_ITERATIONS  100

#define IN_CHANNELS     5
#define OUT_CHANNELS    10

#define IN_HEIGHT       100
#define IN_WIDTH        60

#define OUT_HEIGHT      80
#define OUT_WIDTH       60

#define KERNEL_HEIGHT   8
#define KERNEL_WIDTH    8

#define NUM             64


float *input;
float *output;
float *weight;
float *col_buff;

int conv_in_channels_ = IN_CHANNELS;
int conv_out_channels_ = OUT_CHANNELS;
int conv_in_height_ = IN_HEIGHT;
int conv_in_width_ = IN_WIDTH;
int height_out_ = OUT_HEIGHT;
int width_out_ = OUT_WIDTH;
int kernel_h_ = KERNEL_HEIGHT;
int kernel_w_ = KERNEL_WIDTH;
int num_ = NUM;

int inputSize = conv_in_channels_ * conv_in_height_ * conv_in_width_;
int weightSize = conv_out_channels_ * conv_in_channels_ *
                    kernel_h_ * kernel_w_;
int outputSize = conv_out_channels_ * height_out_ * width_out_;
int colBuffSize = conv_in_channels_ * kernel_w_ * kernel_h_ * conv_in_width_ * conv_in_height_;

void init()
{
  int i;
  float randVal;
  
  int totalInputSize = inputSize * num_;
  int totalOutputSize = outputSize * num_; 
  int totalColBuffSize = colBuffSize * num_;

  input = (float *)malloc(sizeof(float) * totalInputSize);
  if(input == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }

  /* Read random float values */
  for(i = 0; i < totalInputSize; i++) {
    randVal = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    input[i] = randVal;
  }

  weight = (float *)malloc(sizeof(float) * weightSize);
  if(weight == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  /* Read random float values */
  for(i = 0; i < weightSize; i++) {
    randVal = MIN_VAL + static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    weight[i] = randVal;
  }
  
  col_buff = (float *)malloc(sizeof(float) * totalColBuffSize);
  if(col_buff == NULL) {
    printf(" Can