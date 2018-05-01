#!/usr/bin/env python
"""setup.py

Package setup and installation per standard Python practice.

:copyright: Copyright (C) 2014 Perry Sakkaris <psakkaris@gmail.com>
:license: Apache License 2.0, see LICENSE for more details.

http://www.apache.org/licenses/LICENSE-2.0

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from inspect import getfile
from inspect import currentframe
import setuptools
import qudotpy

with open('README.md') as file:
    long_description = file.read()

required_packages = ['numpy']

setuptools.setup(name='qudotpy',
      version='1.0.0',
      description='A quantum computing library written in Python. Can be used to emulate quantum circuits.',
      long_description=long_description,
      keywords='quantum qubit computation emulation qft shor grover',
      author='psakkaris',
      author_email='psakkaris@gmail.com',
      url='https://github.com/psakkaris/qudotpy',
      license='Apache License 2.0',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
				   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Mathematics'
      ],
      platforms='any',
      provides=['qudotpy'],
      data_files=[('', ['README.md',
                        'LICENSE.txt'])],
      namespace_packages=["qudotpy"],
      packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=["numpy"],
      python_requires=">=3.6, < 4",
	  test_suite='qudotpy.test_qudotpy'
)
