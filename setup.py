import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

rootpath = os.path.abspath(os.path.dirname(__file__))


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = [
            "--verbose",
            "--doctest-modules",
            "--ignore",
            "setup.py",
        ]
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


def read(*parts):
    return open(os.path.join(rootpath, *parts)).read()


def extract_version(module="tardis"):
    version = None
    fname = os.path.join(rootpath, module, "__init__.py")
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__"):
                _, version = line.split("=")
                version = version.strip()[1:-1]  # Remove quotation characters.
                break
    return version


email = "ocefpaf@gmail.com"
maintainer = "Filipe Fernandes"
authors = ["Filipe Fernandes"]

LICENSE = read("LICENSE.txt")
long_description = "{}\n{}".format(read("README.rst"), read("CHANGES.txt"))


with open("requirements.txt") as f:
    install_requires = f.readlines()
install_requires = [t.strip() for t in install_requires]


setup(
    name="tardis",
    version=extract_version(),
    packages=["tardis"],
    cmdclass=dict(test=PyTest),
    license=LICENSE,
    long_description=long_description,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
    ],
    description="TARDIS: Collection of functions for SciTools Iris",
    author=authors,
    author_email=email,
    maintainer="Filipe Fernandes",
    maintainer_email=email,
    url="https://github.com/pyoceans/tardis/releases",
    platforms="any",
    keywords=["oceanography", "data analysis", "space-time travel"],
    install_requires=install_requires,
    tests_require="pytest",
    zip_safe=False,
)
