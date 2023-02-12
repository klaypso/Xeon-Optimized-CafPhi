/* Implicit offload of computation on Xeon-Phi 
 * Size chosen based on:
 * VSM size exceeds the limitation (17179869184) now!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

#include "CycleTimer.h"


#define MIN_VAL         (-1)
#define MAX_ITERATIONS  100

#define IN_CHANNELS     3
#define OUT_CHANNELS    10

#define IN_HEIGHT       256
#define IN_WIDTH        256

#define OUT_HEIGHT      80
#define OUT_WIDTH       80

#define KERNEL_HEIGHT   8
#define KERNEL_WIDTH    8

#define NUM             64


typedef struct Convolution {
    int conv_in_channels_;
    int 