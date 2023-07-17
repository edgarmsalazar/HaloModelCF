import os
import glob
import re
from setuptools import setup, find_packages

sources = glob.glob(os.path.join('hmcf','*.py'))

# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


def get_requires():
    reqs = []
    for line in open("requirements.txt", "r").readlines():
        reqs.append(line)
    return reqs

dist = setup(
    name="hmcf",
    autor="Edgar Salazar",
    author_email="edgarmsc@arizona.edu",
    version=get_property("__version__", "hmcf"),
    description="Modules for reproducing Salazar (2023) arXiv:####",
    license="MIT License",
    url="http://github.com/edgarmsalazar/HaloModelCF",
    packages=find_packages(),
    package_data={'hmcf': sources},
    install_requires=get_requires(),
    tests_require=['pytest'],
)