#ifndef HALIDE_EXTRACT_TENSOR_CORE_OPERATIONS_H
#define HALIDE_EXTRACT_TENSOR_CORE_OPERATIONS_H

#include "Expr.h"

namespace Halide {
namespace Internal {
    
Stmt extract_tensor_core_operations(const Stmt& s);

} // namespace Internal
} // namespace Halide

#endif  // HALIDE_EXTRACT_TENSOR_CORE_OPERATIONS_H
