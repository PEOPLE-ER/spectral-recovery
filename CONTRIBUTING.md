- [Contributing](#contributing)
  - [How to Contribute](#how-to-contribute)
    - [Download Code](#download-code)
    - [Installing](#installing)
    - [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)
- [Report Bugs Using GitHub Issues](#report-bugs-using-github-issues)
  - [Bug Reporting Guidelines](#bug-reporting-guidelines)
- [Use a Consistent Coding Style](#use-a-consistent-coding-style)
  - [Python](#python)
- [License](#license)
- [References](#references)
  
# Contributing

We want to make contributing to this project as easy and transparent as possible, whether it is:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## How to Contribute

### Download Code

Make [a fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the main branch of Spectral-Recovery and clone the fork:

```{bash}
git clone https://github.com/<your-github-username>/Spectral-Recovery.git
cd Spectral-Recovery
```

### Installing

From the top-level directory of your Spectral-Recovery clone, you can install the `spectral_recovery` package using pip:

```{bash}
python3 -m pip install -e .
```
This is an "Editable Install" of the spectral_recovery package, meaning you do not have to re-build or re-install each
time you make changes to the source code. See [Development Mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for more information on Development Mode/Editable Installs.

### Running Tests

Spectral-Recovery uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run tests, run the following command from the top-level directory:

```{bash}
pytest src/spectral_recovery
```

Writing tests for your proposed changes is encouraged and will make Pull Request processes smoother!

# Pull Request Process

Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://docs.github.com/en/get-started/quickstart/github-flow)). We actively welcome your pull requests we only ask that you:

1. Ensure your code has been formatted using [black](https://pypi.org/project/black/), and linted using [flake-8](https://flake8.pycqa.org/en/latest/).
   
2. Request the review of two project maintainers. Once two project miantainers have signed-off on the Pull Request it will be merged into the relevant development branch and included in future releases of the spectral_recovery package!

# Report Bugs Using GitHub Issues

We use GitHub Issues to track public bugs. Report a bug by [opening a new issue](https://github.com/PEOPLE-ER/Cloud-Free-Composites/issues/new/choose).

## Bug Reporting Guidelines

Write bug reports with detail, background, and sample code.

**Great Bug Reports** have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Provide a minimal working example showcasing the bug.
- What you expected would happen
- What actually happened
- Notes (possibly including why you think this might be happening, or fixes you tried that didn't work)

We *love* thorough bug reports.

# Use a Consistent Coding Style

## Python 

For all Python code, adhere to PEP8 Standards, to double check your code adheres to PEP8 Standards lint your code before submitting a Pull Request. Do this using [flake8](https://flake8.pycqa.org/en/latest/). 
Function docstrings should be formatted using the [Google Style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). 

# License

By contributing, you agree that your contributions will be licensed under its Apache License.

# References

This document was adapted from the open-source contribution template by [briandk](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).