"""
Phase 7 测试 —— 代码生成、提取、验证。
"""

import pytest


class TestCodeNormalizer:
    def test_extract_cuda_from_fenced_block(self):
        from cuda_opt_agent.codegen.normalizer import extract_cuda_code

        llm_output = '''Here is the code:

```cuda
#include <cuda_runtime.h>

__global__ void kernel(float* data, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) data[i] *= 2.0f;
}
```

This kernel doubles each element.'''

        code = extract_cuda_code(llm_output)
        assert "__global__" in code
        assert "#include" in code
        assert "```" not in code

    def test_extract_cuda_from_cpp_block(self):
        from cuda_opt_agent.codegen.normalizer import extract_cuda_code

        llm_output = '''```cpp
__global__ void k() {}
```'''
        code = extract_cuda_code(llm_output)
        assert "__global__" in code

    def test_extract_bare_code(self):
        from cuda_opt_agent.codegen.normalizer import extract_cuda_code

        bare = '''#include <cuda_runtime.h>
__global__ void k() { }
int main() { return 0; }'''
        code = extract_cuda_code(bare)
        assert "__global__" in code

    def test_extract_longest_block(self):
        from cuda_opt_agent.codegen.normalizer import extract_cuda_code

        llm_output = '''First block:
```cuda
// short
```

Main code:
```cuda
#include <cuda_runtime.h>
__global__ void kernel(float* A, float* B, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) B[i] = A[i] * 2.0f;
}
int main() { return 0; }
```
'''
        code = extract_cuda_code(llm_output)
        assert "main()" in code  # 应取最长的块

    def test_normalize_formatting(self):
        from cuda_opt_agent.codegen.normalizer import normalize_code_formatting

        messy = "line1\n\n\n\nline2\n\n\n\nline3\n"
        clean = normalize_code_formatting(messy)
        # 连续空行合并为一个
        assert "\n\n\n" not in clean
        assert "line1" in clean
        assert "line3" in clean


class TestCodeVerifier:
    def test_valid_code(self, sample_cuda_code):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        result = verify_code_structure(sample_cuda_code)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_empty_code(self):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        result = verify_code_structure("")
        assert result.valid is False
        assert any("空" in e for e in result.errors)

    def test_unmatched_braces(self):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        code = '''
#include <cuda_runtime.h>
__global__ void k() {
    int x = 1;
// missing closing brace
'''
        result = verify_code_structure(code)
        assert result.valid is False
        assert any("闭合" in e or "未闭合" in e for e in result.errors)

    def test_extra_closing_brace(self):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        code = '''
__global__ void k() {
}
}
'''
        result = verify_code_structure(code)
        assert result.valid is False

    def test_no_kernel_warning(self):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        code = '''
#include <stdio.h>
int main() { return 0; }
'''
        result = verify_code_structure(code)
        assert result.valid is True  # 仍然"valid"
        assert any("__global__" in w for w in result.warnings)

    def test_code_with_strings_and_comments(self):
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        code = '''
#include <cuda_runtime.h>
// This is a comment with { and }
__global__ void k() {
    const char* s = "hello { world }";
    /* multi-line
       comment with { braces } */
    int x = 1;
}
'''
        result = verify_code_structure(code)
        assert result.valid is True

    def test_generate_diff(self):
        from cuda_opt_agent.codegen.verifier import generate_diff

        old = "line1\nline2\nline3\n"
        new = "line1\nline2_modified\nline3\nline4\n"
        diff = generate_diff(old, new)
        assert "line2" in diff
        assert "line4" in diff


class TestCodegenIntegration:
    """端到端代码生成流程测试（不调用 LLM）。"""

    def test_extract_then_verify(self):
        from cuda_opt_agent.codegen.normalizer import extract_cuda_code
        from cuda_opt_agent.codegen.verifier import verify_code_structure

        llm_output = '''Here's a simple CUDA kernel:

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int N = 1024;
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    saxpy<<<(N+255)/256, 256>>>(2.0f, d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
```

This implements the SAXPY operation.'''

        code = extract_cuda_code(llm_output)
        result = verify_code_structure(code)
        assert result.valid is True
        assert "__global__" in code
        assert "main()" in code

    def test_layernorm_ref_py_is_unified_cuda_runner(self, tmp_dir):
        import py_compile

        from cuda_opt_agent.codegen.ref_generator import generate_ref_py, ensure_executable_harness

        ref_path = generate_ref_py(
            "layernorm",
            shape_profiles=[{"B": 1024, "N": 1024}],
            default_dtype="fp16",
            output_dir=tmp_dir,
        )
        text = ref_path.read_text(encoding="utf-8")

        assert "--cuda" in text
        assert "def compile_cuda_shared" in text
        assert "def run_cuda_correctness" in text
        assert "def benchmark_cuda_kernel" in text
        assert "def _generate_cuda_inputs" in text
        assert "does not profile PyTorch random-fill kernels" in text
        assert "torch.nn.functional.layer_norm" in text
        assert "int main(" not in text
        py_compile.compile(str(ref_path), doraise=True)

        code = "__global__ void layernorm_smem_tiled_kernel() {}\n"
        assert ensure_executable_harness(code, "layernorm") == code
