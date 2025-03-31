# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import onnx
import numpy as np
import onnx_graphsurgeon as gs
from collections import OrderedDict

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = np.array([80, 280])

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
  return next_node

def simplify_postprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  cls_preds = gs.Variable(name="pred_labels", dtype=np.float32, shape=(1, 80, 280, 18))
  box_preds = gs.Variable(name="pred_boxes", dtype=np.float32, shape=(1, 80, 280, 42))
  dir_cls_preds = gs.Variable(name="pred_scores", dtype=np.float32, shape=(1, 80, 280, 12))

  tmap = graph.tensors()
  new_inputs = [tmap["voxels"], tmap["voxel_coords"], tmap["voxel_num_points"]]
  new_outputs = [cls_preds, box_preds, dir_cls_preds]

  for inp in graph.inputs:
    if inp not in new_inputs:
      inp.outputs.clear()

  for out in graph.outputs:
    out.inputs.clear()

  first_ConvTranspose_node = [node for node in graph.nodes if node.op == "ConvTranspose"][0]
  concat_node = loop_node(graph, first_ConvTranspose_node, 3)
  assert concat_node.op == "Concat"

  first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]]

  for i in range(3):
    transpose_node = loop_node(graph, first_node_after_concat[i], 1)
    assert transpose_node.op == "Transpose"
    transpose_node.outputs = [new_outputs[i]]

  graph.inputs = new_inputs
  graph.outputs = new_outputs
  graph.cleanup().toposort()
  
  return gs.export_onnx(graph)


def simplify_preprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust preprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  MAX_VOXELS = 10000

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, 20, 10))

  # voxel_idxs: [V, 4]
  # V is the maximum number of voxels per frame
  # 4 is just the length of indexs encoded as (frame_id, z, y, x).
  X = gs.Variable(name="voxel_coords", dtype=np.int32, shape=(MAX_VOXELS, 4))

  # voxel_num: [1]
  # Gives valid voxels number for each frame
  Y = gs.Variable(name="voxel_num_points", dtype=np.int32, shape=(1,))
  
  matmul_node_list = [node for node in graph.nodes if node.op == "MatMul"]
  bn_node_list = [node for node in graph.nodes if node.op == "BatchNormalization"]
  relu_node_list = [node for node in graph.nodes if node.op == "Relu"]
  reducemax_node_list = [node for node in graph.nodes if node.op == "ReduceMax"]
  

  # Reshape (for TRT)
  reshape_0 = gs.Node(name="reshape_0", op = "Reshape")
  reshape_0.inputs.append(input_new)
  reshape_0_shape = gs.Constant(name="reshape_0_shape", values = np.array([MAX_VOXELS * 20, 10], dtype=np.int64))
  reshape_0.inputs.append(reshape_0_shape)
  reshape_0_out = gs.Variable(name="reshape_0_out", shape = [MAX_VOXELS * 20, 10], dtype=np.float32)
  reshape_0.outputs.append(reshape_0_out)
  graph.nodes.append(reshape_0)

  # PFN 1st layer matmul 10x32 matmul
  matmul_op_0 = matmul_node_list[0]
  matmul_op_0.inputs[0] = reshape_0_out
  matmul_op_0.inputs[1].name = "vfe.pfn_lyaers.0.linear.weight"
  matmul_op_0_out = gs.Variable(name="matmul_op_0_out", shape = [MAX_VOXELS * 20, 32], dtype=np.float32)
  matmul_op_0.outputs[0] = matmul_op_0_out

  # PFN 1st layer batch norm
  bn_op = bn_node_list[0]
  bn_op.inputs[0] = matmul_op_0_out
  bn_op_out = gs.Variable(name="bn_op_out", shape = [MAX_VOXELS * 20, 32], dtype=np.float32)
  bn_op.outputs[0] = bn_op_out

  # PFN 1st layer relu
  relu_op = relu_node_list[0]
  relu_op.inputs[0] = bn_op_out
  relu_op_out = gs.Variable(name="relu_op_out", shape = [MAX_VOXELS * 20, 32], dtype=np.float32)
  relu_op.outputs[0] = relu_op_out

  # Reshape (for TRT)
  reshape_1 = gs.Node(name="reshape_1", op = "Reshape")
  reshape_1.inputs.append(relu_op_out)
  reshape_1_shape = gs.Constant(name="reshape_1_shape", values = np.array([MAX_VOXELS, 20, 32], dtype=np.int64))
  reshape_1.inputs.append(reshape_1_shape)
  reshape_1_out = gs.Variable(name="reshape_1_out", shape = [MAX_VOXELS, 20, 32], dtype=np.float32)
  reshape_1.outputs.append(reshape_1_out)
  graph.nodes.append(reshape_1)

  # PFN 1st layer reduce max
  reducemax_op = reducemax_node_list[0]
  reducemax_op.inputs[0] = reshape_1_out
  reducemax_op.attrs['keepdims'] = 1
  reducemax_op_out = gs.Variable(name="reducemax_op_out", shape=[MAX_VOXELS, 1, 32], dtype=np.float32)
  reducemax_op.outputs[0] = reducemax_op_out

  # PFN 1st layer tile
  tile_op_0 = gs.Node(name="tile_0", op="Tile")
  tile_op_0.inputs.append(reducemax_op_out)
  tile_op_0_repeat = gs.Constant(name="repeats", values= np.array([1, 20, 1], dtype=np.int64))
  tile_op_0.inputs.append(tile_op_0_repeat)
  tile_op_0_out = gs.Variable(name="tile_0_out", shape=[MAX_VOXELS, 20, 32], dtype=np.float32)
  tile_op_0.outputs.append(tile_op_0_out)
  graph.nodes.append(tile_op_0)

  # Reshape (for TRT)
  reshape_relu = gs.Node(name="reshape_relu", op = "Reshape")
  reshape_relu.inputs.append(relu_op_out)
  reshape_relu_shape = gs.Constant(name="reshape_relu_shape", values = np.array([MAX_VOXELS, 20, 32], dtype=np.int64))
  reshape_relu.inputs.append(reshape_relu_shape)
  reshape_relu_out = gs.Variable(name="reshape_relu_out", shape = [MAX_VOXELS, 20, 32], dtype=np.float32)
  reshape_relu.outputs.append(reshape_relu_out)
  graph.nodes.append(reshape_relu)

  # PFN 1st layer concat
  concat_op_0 = gs.Node(name="concat_0", op="Concat")
  concat_op_0.inputs.append(reshape_relu_out)
  concat_op_0.inputs.append(tile_op_0_out)
  concat_op_0_out = gs.Variable(name="concat_0_out", shape=[MAX_VOXELS, 20, 64], dtype=np.float32)
  concat_op_0.outputs.append(concat_op_0_out)
  concat_op_0.attrs = OrderedDict([('axis', np.int64(-1))])
  graph.nodes.append(concat_op_0)

  ###########################################################
  ###################### PFN 2nd Layer ######################
  ###########################################################

  # Reshape (for TRT)
  reshape_2 = gs.Node(name="reshape_2", op = "Reshape")
  reshape_2.inputs.append(concat_op_0_out)
  reshape_2_shape = gs.Constant(name="reshape_2_shape", values = np.array([MAX_VOXELS * 20, 64], dtype=np.int64))
  reshape_2.inputs.append(reshape_2_shape)
  reshape_2_out = gs.Variable(name="reshape_2_out", shape = [MAX_VOXELS * 20, 64], dtype=np.float32)
  reshape_2.outputs.append(reshape_2_out)
  graph.nodes.append(reshape_2)

  # PFN 2nd layer matmul 64x64 matmul
  matmul_op_1 = matmul_node_list[1]
  matmul_op_1.inputs[0] = reshape_2_out
  matmul_op_1.inputs[1].name = "vfe.pfn_layers.1.linear.weight"
  matmul_op_1_out = gs.Variable(name="matmul_op_1_out", shape=[MAX_VOXELS * 20, 64], dtype=np.float32)
  matmul_op_1.outputs[0] = matmul_op_1_out

  # PFN 2nd layer batch norm
  bn_op_1 = bn_node_list[1]
  bn_op_1.inputs[0] = matmul_op_1_out
  bn_op_1_out = gs.Variable(name="bn_op_1_out", shape = [MAX_VOXELS * 20, 64], dtype=np.float32)
  bn_op_1.outputs[0] = bn_op_1_out

  # PFN 2nd layer relu
  relu_op_1 = relu_node_list[1]
  relu_op_1.inputs[0] = bn_op_1_out
  relu_op_1_out = gs.Variable(name="relu_op_1_out", shape = [MAX_VOXELS * 20, 64], dtype=np.float32)
  relu_op_1.outputs[0] = relu_op_1_out

  reshape_3 = gs.Node(name="reshape_3", op = "Reshape")
  reshape_3.inputs.append(relu_op_1_out)
  reshape_3_shape = gs.Constant(name="reshape_3_shape", values = np.array([MAX_VOXELS, 20, 64], dtype=np.int64))
  reshape_3.inputs.append(reshape_3_shape)
  reshape_3_out = gs.Variable(name="reshape_3_out", shape = [MAX_VOXELS, 20, 64], dtype=np.float32)
  reshape_3.outputs.append(reshape_3_out)
  graph.nodes.append(reshape_3)

  # PFN 2nd layer reduce max
  reducemax_op_1 = reducemax_node_list[1]
  reducemax_op_1.inputs[0] = reshape_3_out
  reducemax_op_1.attrs['keepdims'] = 0
  reducemax_op_1_out = gs.Variable(name="reducemax_op_1_out", shape=[MAX_VOXELS, 1, 64], dtype=np.float32)
  reducemax_op_1.outputs[0] = reducemax_op_1_out

  conv_op = [node for node in graph.nodes if node.op == "Conv"][0]
  graph.replace_with_clip([reducemax_op_1.outputs[0], X, Y], [conv_op.inputs[0]])

  # scatter_op = [node for node in graph.nodes if node.op == "PPScatterPlugin"]
  # print(scatter_op)

  graph.inputs = [input_new, X, Y]
  graph.outputs = [tmap["pred_labels"], tmap["pred_boxes"], tmap["pred_scores"]]

  concat_node_list = [node for node in graph.nodes if node.op == "Concat"]

  graph.cleanup().toposort()
  
  return gs.export_onnx(graph)


if __name__ == '__main__':
    # mode_file = "./model/pointpillars_trt.onnx"
    mode_file = "./model/pointpillar.onnx"
    simplify_preprocess(onnx.load(mode_file))
