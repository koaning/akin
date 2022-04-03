import pathlib
from setuptools import setup, find_packages


base_packages = ["scikit-learn>=1.0.0", "pandas>=1.0.0"]

dev_packages = base_packages + [
    "pytest", 
]


setup(
    name="akin",
    version="0.1.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Gain a clue by clustering!",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/koaning/akin/",
    project_urls={
        "Documentation": "https://github.com/koaning/akin/",
        "Source Code": "https://github.com/koaning/akin/",
        "Issue Tracker": "https://github.com/koaning/akin/issues",
    },
    install_requires=base_packages,
    extras_require={"base": base_packages, "dev": dev_packages},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)