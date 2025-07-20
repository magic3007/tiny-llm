// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "axpby.h"
#include "quantized_matmul.h"
#include "tiny_llm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "bia"_a, "group_size"_a, "bits"_a, "a"_a,
          "b"_a, "transpose_b"_a = false, "stream"_a = nb::none(),
          R"(
        Quantized matmul

        Args:
            scales (array): Scaling factors for ``a``.
            biases (array): Biases for ``a``.
            group_size (int): Group size for ``a``.
            bits (int): Number of bits for ``a``.
            a (array): Input array.
            b (array): Input array.
            transpose_b (bool): Whether to transpose ``b`` before multiplication.
            stream (Stream): Stream to use for the operation.

        Returns:
            array: ``a * b``
      )");
}
