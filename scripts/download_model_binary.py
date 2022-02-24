#!/usr/bin/env python
import os
import sys
import time
import yaml
import urllib
import hashlib
import argparse

required_keys = ['caffemodel', 'caffemodel_url', 'sha1']


def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(