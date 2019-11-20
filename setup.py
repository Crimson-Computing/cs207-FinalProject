import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoDiffCC", # Replace with your own username
    version="0.0.7",
    author="Crimson Computing",
    author_email="majagarbulinska@hsph.harvard.edu",
    description="An AutoDifferentiation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Crimson-Computing/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)