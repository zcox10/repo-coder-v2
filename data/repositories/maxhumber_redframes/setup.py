from setuptools import find_packages, setup

exec(open("redframes/version.py").read())

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="redframes",
    version=__version__,  # type: ignore
    url="https://github.com/maxhumber/redframes",
    description="General Purpose Data Manipulation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Max Humber",
    author_email="max.humber@gmail.com",
    license="BSD 2",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["pandas>=1.5,<2.0"],
    extras_require={
        "test": [
            "matplotlib",
            "scikit-learn",
        ],
        "dev": [
            "black",
            "ipykernel",
            "isort",
            "lxml",
            "matplotlib",
            "mypy",
            "pandas-stubs",
            "pyright",
            "scikit-learn",
            "tabulate",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
