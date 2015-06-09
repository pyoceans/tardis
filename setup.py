# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


rootpath = os.path.abspath(os.path.dirname(__file__))


class PyTest(TestCommand):
    """python setup.py test"""
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--verbose']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


def extract_version():
    version = None
    fname = os.path.join(rootpath, 'tardis', '__init__.py')
    with open(fname) as f:
        for line in f:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters.
                break
    return version


email = "ocefpaf@gmail.com"
maintainer = "Filipe Fernandes"
authors = ['Filipe Fernandes']

LICENSE = read('LICENSE.txt')
long_description = '{}\n{}'.format(read('README.rst'), read('CHANGES.txt'))


# Dependencies.
with open('requirements.txt') as f:
    tests_require = f.readlines()
install_requires = [t.strip() for t in tests_require]


with open('requirements-dev.txt') as f:
    tests_require = f.readlines()
tests_require = [t.strip() for t in tests_require]


config = dict(name='tardis',
              version=extract_version(),
              packages=['tardis'],
              cmdclass=dict(test=PyTest),
              license=LICENSE,
              long_description=long_description,
              classifiers=['Development Status :: 4 - Beta',
                           'Environment :: Console',
                           'Intended Audience :: Science/Research',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Education',
                           'License :: OSI Approved :: MIT License',
                           'Operating System :: OS Independent',
                           'Programming Language :: Python',
                           'Topic :: Education',
                           'Topic :: Scientific/Engineering'],
              description='TARDIS is a collection of functions for Scitools Iris',
              author=authors,
              author_email=email,
              maintainer='Filipe Fernandes',
              maintainer_email=email,
              url='https://github.com/pyoceans/tardis/releases',
              platforms='any',
              keywords=['oceanography', 'data analysis', 'space-time travel'],
              install_requires=install_requires,
              tests_require='pytest',
              zip_safe=False)

setup(**config)
