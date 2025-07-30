#include "quantized_matmul.h"

#include <mlx/array.h>
#include <mlx/primitives.h>

#include "mlx/backend/common/utils.h"  // provide function `elem_to_loc`
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include <Metal/MTLTypes.hpp>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

#include <fstream>
#include <iostream>  // provide function `elem_to_loc`

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Quantized matmul
 *
 *  Args:
 *      scales (array): Scaling factors for ``a``.
 *      biases (array): Biases for ``a``.
 *      group_size (int): Group size for ``a``.
 *      bits (int): Number of bits for ``a``.
 *      a (array): Input array.
 *      b (array): Input array.
 *      transpose_b (bool): Whether to transpose ``b`` before multiplication.
 *      s (StreamOrDevice): Stream or device to use for the operation.
 */
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, const int group_size, const int bits,
                           const mx::array &a, const mx::array &b, const bool transpose_b, mx::StreamOrDevice s) {
    if (scales.dtype() != mx::float16 || biases.dtype() != mx::float16) {
        throw std::runtime_error("quantized_matmul: scales and biases must be float16");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.dtype() != mx::float16) {
        throw std::runtime_error("quantized_matmul: a must be float16");
    }
    if (a.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 64) {
        throw std::runtime_error("quantized_matmul: group_size must be 64");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }

    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b must have the same number of rows as scales");
    }

    if (b.shape()[1] != scales.shape()[1] * group_size / 8) {
        throw std::runtime_error("quantized_matmul: a must have the same number of columns as scales");
    }
    if (a.shape()[1] != b.shape()[1] * 8) {
        throw std::runtime_error("quantized_matmul: a must have the same number of columns as b");
    }

    auto out_shape = a.shape();
    if (out_shape.size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    out_shape[1] = b.shape()[0];

    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ mx::float16,
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <class T>
void quantized_matmul_cpu_launcher(const mx::array &scales, const mx::array &biases, const mx::array &a,
                                   const mx::array &b, T *out_ptr, const mx::Shape &out_shape,
                                   const mx::Strides &out_strides, int group_size) {
    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[0];

    const int bits = 4;
    const int packs_per_item = 32 / bits;

    int group_per_row = N / group_size;
    const T *a_ptr = a.data<T>();
    const uint32_t *b_ptr = b.data<uint32_t>();
    const T *scales_ptr = scales.data<T>();
    const T *biases_ptr = biases.data<T>();

    uint32_t bit_mask = (1 << bits) - 1;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float sum = 0;
            for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                int64_t scales_loc = mx::elem_to_loc(k * group_per_row + group_idx, scales.shape(), scales.strides());
                int64_t biases_loc = mx::elem_to_loc(k * group_per_row + group_idx, biases.shape(), biases.strides());
                int64_t a_loc = mx::elem_to_loc(i * N + group_idx * group_size, a.shape(), a.strides());
                int64_t b_loc =
                    mx::elem_to_loc((k * N + group_idx * group_size) / packs_per_item, b.shape(), b.strides());

                T scale = scales_ptr[scales_loc];
                T bias = biases_ptr[biases_loc];

                for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                    uint32_t b_val = b_ptr[b_loc];
                    // reinterpret_cast用于不相容类型之间的转换，常用用于二进制操作
                    uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val);

                    for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                        uint8_t item_val = (b_bytes[pack_idx >> 1] >> ((pack_idx & 1) * bits)) & bit_mask;
                        float b = item_val * scale + bias;
                        float a = a_ptr[a_loc];
                        sum += a * b;
                        a_loc++;
                    }

                    b_loc++;
                }
            }
            int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
            out_ptr[out_loc] = static_cast<T>(sum);
        }
    }
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());

    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    encoder.dispatch([scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases),
                      a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                      out_ptr = out.data<void>(), out_shape = out.shape(), out_strides = out.strides(),
                      dtype = a.dtype(), group_size = this->group_size_]() {
        if (dtype == mx::float16) {
            quantized_matmul_cpu_launcher(scales, biases, a, b, static_cast<mx::float16_t *>(out_ptr), out_shape,
                                          out_strides, group_size);
        } else if (dtype == mx::bfloat16) {
            quantized_matmul_cpu_launcher(scales, biases, a, b, static_cast<mx::bfloat16_t *>(out_ptr), out_shape,
                                          out_strides, group_size);
        } else {
            throw std::runtime_error("quantized_matmul: unsupported dtype for a");
        }
    });
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    // Make a kernel from this metal library
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("quantized_matmul_w4a16_g64", library);

    auto &encoder = d.get_command_encoder(s.index);

    encoder.set_compute_pipeline_state(kernel);

    // Encode input arrays to kernel
    encoder.set_input_array(scales, 0);
    encoder.set_input_array(biases, 1);
    encoder.set_input_array(a, 2);
    encoder.set_input_array(b, 3);
    // Encode output arrays to kernel
    encoder.set_output_array(out, 4);
    // Encode matrix parameters
    encoder.set_bytes(M, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(K, 7);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = 32;
    const int y_size = tgp_size / x_size;
    if (tgp_size < x_size * y_size) {
        throw std::runtime_error("quantized_matmul: tgp_size must be larger than x*y");
    }

    MTL::Size num_threadgroups = MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1);
    MTL::Size num_threads_per_group = MTL::Size(x_size, y_size, 1);

    encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);
}

}  // namespace tiny_llm_ext