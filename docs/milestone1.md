# Milestone 1 Document
CS207 Final Project, Group 22
Alex Spiride, Maja Garbulinska, Matthew Finney, Zhiying Xu

## Introduction
With the evolution of science from empirical to theoretical and now computational approaches, differentiation plays a critical role in a wide range of scientific and industrial applications of computer science. However, the precise computation of symbolic derivatives is computationally expensive, and not even possible in all situations, whereas the finite differencing method is not always accurate or stable. Automatic Differentiation, however, provides a computationally efficient way to calculate derivatives, particularly of complex functions, for applications where accuracy and performance at scale are important.

## Background


## How to use the package

## Software organization
### Proposed Directory Structure
```
	cs207-FinalProject/
			README.md
			LICENSE
			autodiff/
				dual.py
				math.py
				autodiff.py
				advanced_feature.py [Placeholder]
			docs/
				milestone1.md
			tests/
				test_autodiff.py
			...
```

### Modules
The `autodiff` package will have four modules:

|Module|Basic Functionality|
|-|-|
|autodiff| This is the main module, which will contain the autodiff class and methods for operator overloading (e.g., add, mult, etc.).|
|dual| This module will contain the dual numbers class and methods.|
|math| This module will contain elementary functions, (e.g. sin, cos, sqrt, log, exp,etc.) for the autodiff class. |
|advanced_feature| This module is a placeholder for the advanced feature. Once we've finalized the decision on the advancved feature to build, we will determine whether this needs to be its own module.|

### Test Suite
Our test suite will be in the directory `cs207-FinalProject/tests`. We will use TravisCI to perform continuous integration, running these tests with each build pushed to GitHub. We will also use CodeCov to ensure that our software implementation has sufficient code covered by our test suite. Badges indicating test compliance and code coverage are included in `README.md`.


### Distribution and Packaging
We will distribute our library using the Python Package Index (PyPI). We will distribute the software in the pythonic formats of an sdist and as a wheel to facilitate installation via Python's package installer `pip`. Given that our aim is to deliver software that another developer can use for automatic differentiation in their own applications, we are not planning to deliver a standalone application. As such we are not planning to use a distribution framework beyond Python's native packaging.

## Implementation
