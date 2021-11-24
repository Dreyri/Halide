#include "ExtractTensorCoreOperations.h"

#include "IRMutator.h"
#include "IRVisitor.h"
#include "IROperator.h"
#include "IRPrinter.h"

namespace Halide {
namespace Internal {

namespace {
enum class WMMALayout {
    m16n16k16,
    unknown,
};

WMMALayout get_wmma_layout(int m, int n, int k) {

    std::cout << "Finding layout for M" << m << "N" << n << "K" << k << std::endl;

    if (m == 16 && n == 16 && k == 16) {
        return WMMALayout::m16n16k16;
    }
    
    return WMMALayout::unknown;
}

class CollectForTC : public IRVisitor {
public:
    std::vector<const For*> fors;

public:
    CollectForTC() = default;

    using IRVisitor::visit;

    void visit(const For* op) override {
        internal_assert(op->for_type == ForType::TensorCore) << "Expected Tensor Core dimension\n";
        fors.push_back(op);
        IRVisitor::visit(op);
    }
};

std::vector<const For*> collect_for_tc(Stmt s) {
    CollectForTC collect{};
    s.accept(&collect);
    return std::move(collect.fors);
}

class CollectLetStmts : public IRVisitor {
public:
    std::vector<const LetStmt*> lets;

public:
    CollectLetStmts() = default;

    using IRVisitor::visit;

    void visit(const LetStmt* op) override {
        lets.push_back(op);
    }
};

std::vector<const LetStmt*> collect_lets(Stmt s) {
    CollectLetStmts collector{};
    s.accept(&collector);
    return std::move(collector.lets);
}

int64_t get_const_extent(const For* op) {
    internal_assert(is_const_zero(op->min) && is_const(op->extent));

    return op->extent.as<IntImm>()->value;
}

class ExtractTensorCoreOperations : public IRMutator {

    std::vector<const LetStmt*> lets;

    const For* m_dim;
    const For* n_dim;
    const For* k_dim;

    int m_extent{-1};
    int n_extent{-1};
    int k_extent{-1};

public:
    ExtractTensorCoreOperations() = default;

    using IRMutator::visit;

    Stmt visit(const For* loop) override {
        if (loop->for_type == ForType::GPUBlock) {
            IRPrinter p{std::cout};
            loop->accept(&p);
            std::cout << std::endl;
        }

        if (loop->for_type == ForType::TensorCore) {
            // we're now where we want to start mutating to get tensor core operations

            IRPrinter p{std::cout};
            loop->accept(&p);
            std::cout << std::endl;

            m_dim = loop;
            m_extent = get_const_extent(loop);

            // gather all let statements we can find so we can recover offset and stride later
            lets = collect_lets(loop);

            auto tc_fors = collect_for_tc(m_dim->body);
            internal_assert(tc_fors.size() == 2) << "Expected 3 Tensor Core dimensions, got " << tc_fors.size() + 1 << "\n";
            n_dim = tc_fors[0];
            n_extent = get_const_extent(n_dim);

            k_dim = tc_fors[1];
            k_extent = get_const_extent(k_dim);

            std::cout << "Found the required loop nesting" << std::endl;

            return Stmt{};
        }

        return IRMutator::visit(loop);
    }
};
}

Stmt extract_tensor_core_operations(const Stmt& s) {
    return ExtractTensorCoreOperations{}.mutate(s);
}
}
}