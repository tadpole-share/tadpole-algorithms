import setuptools
import codecs
import os

# Collect requirements from all requirements.txt files
install_requires = set()
for root,dirs,files in os.walk('.'):
    if 'requirements.txt' in files:
        reqs_file = os.path.join(root, 'requirements.txt')
        with codecs.open(reqs_file, mode='r') as f:
            reqs = f.read().splitlines()
            install_requires.update(reqs)

setuptools.setup(
    name="tadpole_algorithms",
    version="1.0.0",
    author="Eyra team",
    author_email="c.martinez@esciencecenter.nl",
    description="Tadpole algorithms",
    long_description="Tadpole algorithms",
    long_description_content_type="text/markdown",
    url="https://github.com/tadpole-share/tadpole-algorithms",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=list(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)
