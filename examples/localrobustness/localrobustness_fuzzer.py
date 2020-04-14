"""Fuzz a neural network for local robustness."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import onnx
import os
import random
import tensorflow as tf

from onnx import numpy_helper

from tensorfuzz.fuzz_utils import build_fetch_function
from tensorfuzz.corpus import InputCorpus
from tensorfuzz.corpus import seed_corpus_from_numpy_arrays
from tensorfuzz.coverage_functions import raw_logit_coverage_function
from tensorfuzz.fuzzer import Fuzzer
from tensorfuzz.mutation_functions import do_basic_mutations
from tensorfuzz.sample_functions import recent_sample_function


tf.flags.DEFINE_string("model", None, "Model to fuzz.")
tf.flags.DEFINE_string("lb", None, "Lower bound of input.")
tf.flags.DEFINE_string("ub", None, "Upper bound of input.")
tf.flags.DEFINE_integer("label", None, "Desired label.")
tf.flags.DEFINE_integer("seed", None, "Random seed for both python and numpy.")
tf.flags.DEFINE_integer("total_inputs_to_fuzz", 100, "Loops over the whole corpus.")
tf.flags.DEFINE_integer(
    "mutations_per_corpus_item", 100, "Number of times to mutate corpus item."
)
tf.flags.DEFINE_float(
    "ann_threshold", 1.0, "Distance below which we consider something new coverage.",
)
FLAGS = tf.flags.FLAGS

ONNX_TO_NUMPY_DTYPE = {
    onnx.TensorProto.DOUBLE: np.dtype("float64"),
    onnx.TensorProto.FLOAT16: np.dtype("float16"),
    onnx.TensorProto.FLOAT: np.dtype("float32"),
    onnx.TensorProto.INT16: np.dtype("int16"),
    onnx.TensorProto.INT32: np.dtype("int32"),
    onnx.TensorProto.INT64: np.dtype("int64"),
}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    elif isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    elif isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return np.float(node.f)
        elif node.type == onnx.AttributeProto.INT:
            return np.int(node.i)
        elif node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        elif node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError("Unknown attribute type: %s" % (node,))
    else:
        raise ValueError("Unknown node type: %s" % type(node))


def metadata_function(metadata_batches):
    """Gets the metadata."""
    metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
    return metadata_list


def objective_function(corpus_element):
    """Checks if the predicted output class is different then specified."""
    y = corpus_element.metadata[0]
    if any(y[FLAGS.label] <= y[i] for i in range(len(y)) if i != FLAGS.label):
        tf.logging.info("Objective function satisfied: differently classified input found.")
        return True
    return False


def onnx_to_tf(node, *inputs):
    if isinstance(node, onnx.ValueInfoProto):
        dims = [
            -1 if dim.dim_param else dim.dim_value
            for dim in node.type.tensor_type.shape.dim
        ]
        dims[0] = None
        shape = np.array(dims)
        dtype = ONNX_TO_NUMPY_DTYPE[node.type.tensor_type.elem_type]
        return tf.placeholder(dtype, shape)
    elif not isinstance(node, onnx.NodeProto):
        raise ValueError("Unknown node type: %s" % type(node))
    if node.op_type == "Conv":
        assert len(inputs) == 3
        x, w, b = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        dilations = list(attributes.get("dilations", [1, 1]))
        pads = list(attributes.get("pads", [0, 0, 0, 0]))
        strides = list(attributes.get("strides", 1))
        x_padded = tf.pad(
            x,
            [(0, 0), (0, 0)]
            + list(zip(pads[: len(pads) // 2], pads[len(pads) // 2 :])),
        )
        x_padded_T = tf.transpose(x_padded, (0, 2, 3, 1))
        y_T = tf.nn.bias_add(
            tf.nn.conv2d(
                x_padded_T,
                w.transpose(2, 3, 1, 0),
                [1] + strides + [1],
                "VALID",
                dilations=[1] + dilations + [1],
            ),
            b,
        )
        return tf.transpose(y_T, (0, 3, 1, 2))
    elif node.op_type == "Relu":
        assert len(inputs) == 1
        [x] = inputs
        return tf.nn.relu(x)
    elif node.op_type == "Sigmoid":
        assert len(inputs) == 1
        [x] = inputs
        return tf.nn.sigmoid(x)
    elif node.op_type == "Tanh":
        assert len(inputs) == 1
        [x] = inputs
        return tf.nn.tanh(x)
    elif node.op_type == "Transpose":
        assert len(inputs) == 1
        [x] = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        perm = attributes.get("perm")
        return tf.transpose(x, perm)
    elif node.op_type == "Reshape":
        assert len(inputs) == 2
        x, shape = inputs
        return tf.reshape(x, shape)
    elif node.op_type == "Gemm":
        assert len(inputs) == 3
        a, b, c = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        alpha = attributes.get("alpha", 1.0)
        beta = attributes.get("beta", 1.0)
        transA = bool(attributes.get("transA", False))
        transB = bool(attributes.get("transB", False))
        return (
            alpha * tf.matmul(a, b, transpose_a=transA, transpose_b=transB,) + beta * c
        )
    elif node.op_type == "MatMul":
        assert len(inputs) == 2
        a, b = inputs
        return tf.matmul(a, b)
    elif node.op_type == "Add":
        assert len(inputs) == 2
        a, b = inputs
        return a + b
    elif node.op_type == "Pad":
        assert len(inputs) == 1
        [x] = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        mode = attributes.get("mode", "constant")
        pads = list(attributes.get("pads"))
        value = attributes.get("value", 0.0)
        return tf.pad(
            x,
            list(zip(pads[: len(pads) // 2], pads[len(pads) // 2 :])),
        )
    elif node.op_type == "Flatten":
        assert len(inputs) == 1
        [x] = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        axis = attributes.get("axis", 1)
        if axis == 0:
            new_shape = (1, -1)
        elif axis == 1:
            new_shape = (-1, int(np.prod(x.shape[1:])))
        else:
            new_shape = (int(np.prod(x.shape[:axis])), -1)
        return tf.reshape(x, new_shape)
    elif node.op_type == "Concat":
        assert len(inputs) == 1
        [x] = inputs
        attributes = {a.name: as_numpy(a) for a in node.attribute}
        axis = attributes.get("axis")
        return tf.concat(x, axis=axis)
    raise NotImplementedError("Unsupported op type: %s" % node.op_type)


def get_tensors_from_onnx_model(sess, path):
    # BEGIN PARSING
    onnx_model = onnx.load(path)
    node_map = {}
    operation_map = {}
    parameter_map = {}
    for node in onnx_model.graph.node:
        if node.op_type in ["Constant"]:
            assert len(node.output) == 1
            parameter_map[node.output[0]] = as_numpy(node)
        else:
            for output_name in node.output:
                node_map[output_name] = node
    for initializer in onnx_model.graph.initializer:
        parameter_map[initializer.name] = as_numpy(initializer)
    for input_node in onnx_model.graph.input:
        if input_node.name not in parameter_map:
            operation_map[input_node.name] = onnx_to_tf(input_node)

    operations = []
    visited = set()  # type: Set[int]

    def topo_sort(node):
        if id(node) in visited:
            return operation_map[id(node)]
        visited.add(id(node))
        inputs = []
        for name in node.input:
            if name in node_map:
                topo_sort(node_map[name])
                inputs.append(operation_map[name])
            elif name in parameter_map:
                inputs.append(parameter_map[name])
            elif name in operation_map:
                inputs.append(operation_map[name])
            else:
                raise ValueError("Unknown input name: %s" % name)
        operation = onnx_to_tf(node, *inputs)
        if len(node.output) > 1:
            for i, output_name in enumerate(node.output):
                operation_map[output_name] = operation[i]
        else:
            operation_map[node.output[0]] = operation
        operation_map[id(node)] = operation
        operations.append(operation)

    for node in node_map.values():
        topo_sort(node)

    output_operations = []
    for output_info in onnx_model.graph.output:
        output_operations.append(operation_map[output_info.name])
    # END PARSING

    input_tensors = []
    for input_node in onnx_model.graph.input:
        if input_node.name not in parameter_map:
            input_tensors.append(operation_map[input_node.name])
    input_tensors.append(tf.placeholder(tf.int64, (None,)))
    coverage_tensors = []
    for output_node in onnx_model.graph.output:
        coverage_tensors.append(operation_map[output_node.name])
        # coverage_tensors.append(tf.nn.softmax(operation_map[output_node.name]))
    metadata_tensors = []
    for output_node in onnx_model.graph.output:
        metadata_tensors.append(operation_map[output_node.name])

    tensor_map = {
        "input": input_tensors,
        "coverage": coverage_tensors,
        "metadata": metadata_tensors,
    }
    return tensor_map


def main(args):
    """Configures and runs the fuzzer."""
    # Log more
    tf.logging.set_verbosity(tf.logging.INFO)
    # Set the seeds!
    if FLAGS.seed:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
    
    if FLAGS.label is None:
        raise ValueError("expected label must be provided")

    lower_bound = np.zeros((1, 28, 28))
    upper_bound = np.ones((1, 28, 28))
    if FLAGS.lb is not None:
        lower_bound = np.load(FLAGS.lb)
    if FLAGS.ub is not None:
        upper_bound = np.load(FLAGS.ub)

    coverage_function = raw_logit_coverage_function
    image = (lower_bound + upper_bound) / 2
    numpy_arrays = [[image, -1]]

    with tf.Session() as sess:
        tensor_map = get_tensors_from_onnx_model(sess, FLAGS.model)

        fetch_function = build_fetch_function(sess, tensor_map)

        size = FLAGS.mutations_per_corpus_item
        mutation_function = lambda elt: do_basic_mutations(
            elt, size, a_min=lower_bound, a_max=upper_bound
        )

        seed_corpus = seed_corpus_from_numpy_arrays(
            numpy_arrays, coverage_function, metadata_function, fetch_function
        )
        corpus = InputCorpus(
            seed_corpus, recent_sample_function, FLAGS.ann_threshold, "kdtree"
        )

        fuzzer = Fuzzer(
            corpus,
            coverage_function,
            metadata_function,
            objective_function,
            mutation_function,
            fetch_function,
        )
        result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)
        if result is not None:
            tf.logging.info("Fuzzing succeeded.")
            tf.logging.info(
                "Generations to make satisfying element: %s.",
                result.oldest_ancestor()[1],
            )
            y = sess.run(tensor_map["metadata"], feed_dict={tensor_map["input"][0]: np.expand_dims(result.data[0], 0)})
            print(y, np.argmax(y))
            result_filename = os.path.join(os.path.dirname(FLAGS.model), "cex.npy")
            np.save(result_filename, result.data[0])
        else:
            tf.logging.info("Fuzzing failed to satisfy objective function.")


if __name__ == "__main__":
    tf.app.run()
