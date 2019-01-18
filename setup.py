import setuptools


LONG_DESCRIPTION = """
Tools for computing derivatives on the sphere in xarray.
"""


setuptools.setup(
    name='xdiff',
    version='0.1',
    packages=setuptools.find_packages(),
    author='Spencer K. Clark',
    description='General differentiation tools',
    long_description=LONG_DESCRIPTION,
    install_requires=[
        'numpy >= 1.7',
        'xarray >= 0.11',
    ],
)
