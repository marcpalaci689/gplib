from setuptools import setup

setup(name='gplib',
      version='0.0.1',
      description='Python Library for efficient Gaussian Process models',
      author='Marc Palaci-Olgun',
      author_email='marcpalaci689@gmail.com',
      packages = ['gplib','gplib/core','gplib/models','gplib/kernels','gplib/grid'])