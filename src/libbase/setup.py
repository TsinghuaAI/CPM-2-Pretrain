import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os


if __name__ == '__main__':
    setuptools.setup(
        name='libbase',
        version='0.1.0',
        description='libbase',
        author='fair',
        author_email='fair',
        ext_modules=[
            CppExtension(
                "libbase",
                sources=[
                    "balanced_assignment.cpp",
                ],
            )
            ],
        cmdclass={
            'build_ext': BuildExtension
        })
