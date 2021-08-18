"""Package configuration."""

from setuptools import setup


def _readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="multimodemodel",
    version="0.1",
    description="Nonlinear Multimode model",
    long_description=_readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        # 'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3.7",
        # 'Topic :: Text Processing :: Linguistic',
    ],
    url="https://git.geomar.de/mcgroup/multimode-model",
    author="Martin Claus",
    author_email="mclaus@geomar.de",
    # license='MIT',
    packages=["multimodemodel"],
    python_requires=">=3.7",
    install_requires=[
        "numba >= 0.50.1",
        "numpy",
    ],
    extras_require={
        "xarray": ["xarray"],
    }
    # zip_safe=False
)
