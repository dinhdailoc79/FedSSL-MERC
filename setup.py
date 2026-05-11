"""
FedSSL-MERC: Project Configuration
====================================
"""
from setuptools import setup, find_packages

setup(
    name="fedssl-merc",
    version="0.1.0",
    author="Dinh Dai Loc",
    author_email="dinhdailoc79@gmail.com",
    description="Federated Semi-Supervised Learning for Multimodal Emotion Recognition in Conversations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dinhdailoc79/FedSSL-MERC",
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
