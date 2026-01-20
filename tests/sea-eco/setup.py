from __future__ import print_function
from setuptools import setup


setup(name='sea-eco',
      version='1.0',
      description='Useful functions for python. Mostly personal use.',
      author='Eric Hoglund',
      author_email='hoglunder@ornl.gov',
      packages=['pySEA/sea_eco'],
      install_requires=['numpy', 'matplotlib','hyperspy>=2.0']
     )
