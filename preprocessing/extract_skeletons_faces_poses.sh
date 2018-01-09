#!/bin/bash

EXPECTED_ARGS=0
E_BADARGS=65
FRAMES=30

if [ $# -lt $EXPECTED_ARGS ]
then
  exit $E_BADARGS
fi

SAVEPATH="../data/group_Skeleton_face_pose"
mkdir -m 755 $SAVEPATH

./openpose-master/build/examples/openpose/openpose.bin --face --image_dir "../data/Test_converted" --num_gpu 1 --write_images $SAVEPATH --disable_blending




