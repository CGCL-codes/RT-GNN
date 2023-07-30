from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='Fusion',
    ext_modules=[
        CUDAExtension('Fusion', [
            'GCNFusion.cpp',
            'GCNFusion_V100.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })