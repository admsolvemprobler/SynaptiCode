from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="synapticode",
    version="0.1.0",
    author="SynaptiCode Team",
    author_email="info@synapticode.example.com",
    description="A Living Codebase System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/admsolvemprobler/SynaptiCode",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "openai>=0.27.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "synapticode=src.cli:main",
        ],
    },
)
