import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = ['src/dcn_v2.c']
headers = ['src/dcn_v2.h']
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/dcn_v2_cuda.c']
    headers += ['src/dcn_v2_cuda.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/dcn_v2_im2col_cuda.cu.o']
    extra_objects += ['src/cuda/dcn_v2_psroi_pooling_cuda.cu.o']
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = ['-fopenmp', '-std=c99']

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ccp_ext = CppExtension(
    name='_ext.dcn_v2',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
)

if __name__ == '__main__':
    setup(name='_ext.dcn_v2',
          ext_modules=[
              ccp_ext
          ],
          cmdclass={
              'build_ext': BuildExtension
          })
