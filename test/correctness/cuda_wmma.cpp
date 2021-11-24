#include "Halide.h"

using namespace Halide;

float16_t calc_a(int x, int y) {
    return float16_t(static_cast<float>(x - y * 17));
}

float16_t calc_b(int x, int y) {
    return float16_t(static_cast<float>(x * 3 + y * 7));
}

void test_wmma(Target t) {

    Var i("i"), j("j"), ii("ii"), ji("ji");
    RVar ki("ki"), kii("kii"), ko("ko");

    Func A("A"), B("B");

    A(i, j) = cast<float16_t>(i - j * 17);
    B(i, j) = cast<float16_t>(i * 3 + j * 7);

    A.compute_root();
    B.compute_root();

    int k_ = 128;

    RDom k{0, k_};

    Func C("C");
    C(i, j) = cast<float>(i + j);
    C(i, j) += cast<float>(A(k, j)) * cast<float>(B(i, k));

    C.compute_root()
        .store_in(MemoryType::Stack)
        .update()
        .tile(i, j, ii, ji, 16, 16)
        .split(k, ko, ki, 16)
        .atomic()
        .gpu_blocks(i, j)
        .reorder({ki, ii, ji, ko, i, j})
        .tensor_core(ii, ji, ki);


    int m = 128;
    int n = 128;

    Buffer<float> output(m, n);

    //C.compile_to_lowered_stmt("tc.html", {}, HTML, t);
    C.print_loop_nest();
    C.realize(output, t);

    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < n; ++x) {
            float val = 0.f;
            for (int z = 0; z < k_; ++z) {
                val += static_cast<float>(calc_a(z, y)) * static_cast<float>(calc_b(x, z));
            }

            if (output(x, y) != val) {
                std::cout << "output(" << x << ", " << y << ") = " << output(x, y) << " instead of " << val << std::endl;
            }
        }
    }
}

int main(int argc, char** argv) {
    Target t = get_jit_target_from_environment();
    if (!t.has_feature(Target::CUDACapability70)) {
        printf("[SKIP] Cuda (with compute capability 7.0) is not enabled in target: %s\n",
               t.to_string().c_str());
        return 0;
    }

    test_wmma(t);

    printf("Success!\n");
    return 0;
}