from glob import glob
import os
import sys
from setuptools import setup


name = "pypdc"
version="0.0.5"
description = "Python asymptotic Partial Directed Coherence and Directed Coherence estimation package for brain connectivity analysis."
authors = {
    "Sameshima": ("Koichi Sameshima", "ksameshi@usp.br"),
    "Brito": ("Carlos Stein Naves de Brito", "c.brito@ucl.ac.uk"),
    "Baldo" : ("Heitor Baldo", "hbaldo@usp.br")
}

platforms = ["Linux", "Mac OSX", "Windows", "Unix"]
keywords = [
    "Brain Connectivity",
    "PDC", "iPDC", 
    "Granger Causality",
    ]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]


packages = ["pypdc"]
    

with open("README.rst", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    setup(
        name=name,
        version=version,
        author=authors["Sameshima"][0],
        author_email=authors["Sameshima"][1],
        description=description,
        keywords=keywords,
        platforms=platforms,
        classifiers=classifiers,
        packages=packages,
        zip_safe=False,
    )