#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
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
                           const mx::array &a, const mx::array &b, const bool transpose_b, mx::StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(mx::Stream stream, int group_size, int bits)
        : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    void print(std::ostream &os) override { os << "QuantizedMatmul"; }

private:
    int group_size_;
    int bits_;
};

}  // namespace tiny_llm_ext