# -*- coding: utf-8 -*-
 
 
"""setup.py: setuptools control."""
 
 
import re
from setuptools import setup
 
 
version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('astrotools/__init__.py').read(),
    re.M
    ).group(1)
 
 
with open("README.rst", "rb") as f:
    long_descr = f.read().decode("utf-8")
 
 
setup(
    name = "cjhang-spectools",
    packages = ["spectools"],
    #entry_points = {
    #    "console_scripts": ['bootstrap = bootstrap.bootstrap:main']
    #    },
    version = version,
    description = "Useful astronomy tools",
    long_description = long_descr,
    author = "Jianhang Chen",
    author_email = "cjhastro@gmail.com",
    url = "http://github.com/cjhang/astrotools.git",
    )
