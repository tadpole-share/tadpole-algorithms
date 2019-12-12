import setuptools

setuptools.setup(
    name="tadpole_algorithms",
    version="1.0.0",
    author="Eyra team",
    author_email="t.klaver@esciencecenter.nl",
    description="Tadpole algorithms",
    long_description="Tadpole algorithms",
    long_description_content_type="text/markdown",
    url="https://github.com/tadpole-share/tadpole-algorithms",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
      'numpy',
      'pandas',
      'scikit-learn',
      'scipy',
      'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)
