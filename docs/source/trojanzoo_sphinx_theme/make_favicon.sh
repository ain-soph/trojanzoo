#!/bin/bash -xe
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Pytorch hex color: #ee4c2c

IMG_DIR=./static/images/

convert -resize 32 -background none $IMG_DIR/logo-icon.svg $IMG_DIR/favicon.ico