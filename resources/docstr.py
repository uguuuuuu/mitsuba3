import subprocess
from glob import glob

subprocess.run(['rm', 'docstr.h'])

file_list  = glob("include/mitsuba/core/**/*.h", recursive=True)
file_list += glob("include/mitsuba/render/**/*.h", recursive=True)

args = ['pybind11-mkdoc', '-o', 'docstr.h',
        # include directories 
        '-I', 'include', '-I', 'build/include', 
        '-I', 'ext/drjit/include', '-I', 'ext/drjit/ext/drjit-core/include',
        '-I', 'ext/drjit/ext/drjit-core/ext/nanothread/include',
        '-I', 'ext/asmjit/src', '-I', 'ext/embree/include', '-I', 'ext/tinyformat',
        # macros
        '-D', 'MI_ENABLE_CUDA', '-D', 'MI_ENABLE_EMBREE', '-D', 'MI_ENABLE_AUTODIFF',
        # specify cpp standard
        '-std=c++17']
# headers to process
args += file_list

process = subprocess.run(args, stdout=subprocess.PIPE)