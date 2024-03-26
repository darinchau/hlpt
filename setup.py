from setuptools import find_packages, setup

VERSION = "0.1.0"

setup(
    name="mlutils",
    version=VERSION,
    description="A collection of utilities for deep learning models",
    license_files=["LICENSE"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="MIT",
    author="Darin Chau",
    author_email="darinchauyf@gmail.com",
    url="https://github.com/darinchau/mlutils",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
