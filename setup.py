from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simple_flash_attn',
    ext_modules=[
        CUDAExtension(
            'simple_flash_attn_cuda',
            [
                'simple_flash_attn/flash_api.cpp',
                'simple_flash_attn/flash_api.cu',
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-arch=sm_80'] # sm_80 (Ampere) required for cp.async (cuda::pipeline) hardware support
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
