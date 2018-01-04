"""
Package meta-data.
"""

from setuptools import setup

setup(
    name='cnn-toys',
    version='0.0.1',
    description='Playing around with CNNs.',
    url='https://github.com/unixpickle/cnn-toys',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='MIT',
    packages=['cnn_toys'],
    install_requires=['tensorflow', 'numpy']
)
