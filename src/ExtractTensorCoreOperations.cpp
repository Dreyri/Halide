#include "ExtractTensorCoreOperations.h"

#include "IRMatch.h"
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
    std::vector<const LetStmt*>* lets;

public:
    CollectLetStmts(std::vector<const LetStmt*>& lets)
        : lets(&lets)
    {}

    using IRVisitor::visit;

    void visit(const LetStmt* op) override {
        lets->push_back(op);
        IRVisitor::visit(op);
    }
};

void collect_lets(Stmt s, std::vector<const LetStmt*>& lets) {
    CollectLetStmts collector{lets};
    s.accept(&collector);
}

int64_t get_const_extent(const For* op) {
    internal_assert(is_const_zero(op->min) && is_const(op->extent));

    return op->extent.as<IntImm>()->value;
}

class ExtractStore : public IRVisitor {
public:
    const Store* op;

    ExtractStore() = default;

    using IRVisitor::visit;

    void visit(const Store* op) override {
        this->op = op;
        return;
    }
};

class ReplaceVars : public IRMutator {
    const std::vector<const LetStmt*>* lets;
public:
    ReplaceVars(const std::vector<const LetStmt*>& lets)
        : lets(&lets)
    {}

    using IRMutator::visit;

    const LetStmt* find_let(const std::string& name) const {
        auto it = std::find_if(lets->rbegin(), lets->rend(), [&](const LetStmt* let) {
            return let->name == name;
        });

        if (it == lets->rend()) {
            return nullptr;
        }

        return *it;
    }

    Expr visit(const Variable* op) override {
        const LetStmt* let = find_let(op->name);

        if (!let) {
            return IRMutator::visit(op);
        }

        return mutate(let->value);
    }
};

enum struct MatrixLoadType {
    Row,
    Col,
    Unknown,
};

struct MatrixLoads {
    const Store* c_store{nullptr};
    const Load* c_load{nullptr};
    const Load* a_load{nullptr};
    const Load* b_load{nullptr};

    explicit operator bool() const {
        return c_store && c_load && a_load && b_load;
    }

    // 0 for row, 1 col, -1 invalid
    int identify_matrix_load_a(const For* m_dim, const For* n_dim, const For* k_dim) const {
        auto k_var = Variable::make(Int(32), k_dim->name);
        auto m_var = Variable::make(Int(32), m_dim->name);
        auto wildcard = Variable::make(Int(32), "*");

        std::vector<Expr> match_results;

        Expr a_row_pattern = k_var + ((((m_var + wildcard) * 8) + wildcard) * 16);

        if (expr_match(a_row_pattern, a_load->index, match_results)) {
            return 0;
        }

        Expr a_col_pattern = ((k_var + wildcard)*(wildcard + 16)) + (m_var + wildcard);
        if (expr_match(a_col_pattern, a_load->index, match_results)) {
            return 1;
        }

        return -1;
    }

    int identify_matrix_load_b(const For* m_dim, const For* n_dim, const For* k_dim) const {
        
        auto k_var = Variable::make(Int(32), k_dim->name);
        auto n_var = Variable::make(Int(32), n_dim->name);
        auto wildcard = Variable::make(Int(32), "*");

        std::vector<Expr> match_results;

        Expr b_row_pattern = ((k_var + wildcard) * (wildcard + 16)) + (n_var + wildcard);

        if (expr_match(b_row_pattern, b_load->index, match_results)) {
            return 0;
        }

        Expr b_col_pattern = k_var + ((((n_var + wildcard) * 8) + wildcard) * 16);

        if (expr_match(b_row_pattern, b_load->index, match_results)) {
            return 1;
        }

        return -1;
    }

    MatrixLoadType identify_matrix_load(const For* m_dim, const For* n_dim, const For* k_dim) const {
        auto k_var = Variable::make(Int(32), k_dim->name);
        auto m_var = Variable::make(Int(32), m_dim->name);
        auto n_var = Variable::make(Int(32), n_dim->name);
        auto idx_wc = Variable::make(Int(32), "*");
        auto c_stride1_var = Variable::make(Int(32), c_load->name + ".stride.1");
        auto c_stride0_var = Variable::make(Int(32), c_load->name + ".stride.0");

        std::vector<Expr> match_results;

        // This isn't the most reliable and can easily fail matching if the order of operations is different
        Expr c_row_pattern = (n_var + (((m_var + idx_wc) * c_stride1_var) + idx_wc));

        std::cout << "Trying to match the following pattern\n";
        IRPrinter p{std::cout};
        c_load->index.accept(&p);
        std::cout << std::endl;
        std::cout << "Using the following pattern\n";
        c_row_pattern.accept(&p);
        std::cout << std::endl;

        if (!expr_match(c_row_pattern, c_load->index, match_results)) {
            return MatrixLoadType::Unknown;
        }

        auto a_pattern = identify_matrix_load_a(m_dim, n_dim, k_dim);


        Expr b_row_pattern = ((k_var + idx_wc) * (idx_wc + 16)) + (n_var + idx_wc);

        if (!expr_match(b_row_pattern, b_load->index, match_results)) {
            std::cout << "B load is not a row load" << std::endl;
        }

    }
};

MatrixLoads extract_matrix_loads(const Store* op) {
    MatrixLoads indices;

    indices.c_store = op;

    const auto wildf32 = Variable::make(Float(32, 1), "*");

    Expr pattern = wildf32 + wildf32 * wildf32;

    std::vector<Expr> match_results;
    if (!expr_match(pattern, op->value, match_results)) {
        return {};
    }

    IRPrinter p{std::cout};

    std::cout << "C\n";
    match_results[0].accept(&p);
    std::cout << std::endl;

    std::cout << "A\n";
    match_results[1].accept(&p);
    std::cout << std::endl;

    std::cout << "B\n";
    match_results[2].accept(&p);
    std::cout << std::endl;

    const Load* load_c = match_results[0].as<Load>();
    const Cast* load_a_cast = match_results[1].as<Cast>();
    const Cast* load_b_cast = match_results[2].as<Cast>();

    if (!load_a_cast || !load_b_cast) {
        return {};
    }

    const Load* load_a = load_a_cast->value.as<Load>();
    const Load* load_b = load_b_cast->value.as<Load>();

    indices.c_load = load_c;
    indices.a_load = load_a;
    indices.b_load = load_b;

    return indices;
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
            collect_lets(loop, lets);

            auto tc_fors = collect_for_tc(m_dim->body);
            internal_assert(tc_fors.size() == 2) << "Expected 3 Tensor Core dimensions, got " << tc_fors.size() + 1 << "\n";
            n_dim = tc_fors[0];
            n_extent = get_const_extent(n_dim);

            k_dim = tc_fors[1];
            k_extent = get_const_extent(k_dim);

            std::cout << "Found the required loop nesting" << std::endl;

            WMMALayout layout = get_wmma_layout(m_extent, n_extent, k_extent);

            internal_assert(layout != WMMALayout::unknown) << "Got unknown Tensor Core layout: M" << m_extent << "N" << n_extent << "K" << k_extent << "\n";

            ExtractStore get_store{};

            k_dim->accept(&get_store);

            const Store* store_c = get_store.op;

            std::cout << "Store C\n";
            store_c->accept(&p);
            std::cout << std::endl;

            ReplaceVars replacer{lets};
            Stmt store_c2 = replacer.mutate(store_c);

            std::cout << "C after replacing\n";
            store_c2.accept(&p);
            std::cout << std::endl;

            MatrixLoads loads = extract_matrix_loads(store_c2.as<Store>());

            internal_assert(loads) << "Could not recognize the load pattern required for WMMA\n";

            MatrixLoadType load_type = loads.identify_matrix_load(m_dim, n_dim, k_dim);

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