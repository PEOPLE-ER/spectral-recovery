# Installation

## Stable Release

spectral_recovery and all of it's required dependencies can be installed using pip (or uv):

```bash
pip install spectral_recovery
```

To install the CLI version of spectral_recovery, use "spectral_recovery[cli]" instead:

```bash
pip install "spectral_recovery[cli]"
```

## From Source

> [!WARNING]
> Unreleased source code is not considered stable and might even contain undetected bugs. Proceed with caution if using source code in your workflows.

spectral_recovery can be installed from source code directly from the project repository:

```bash
pip install git+https://github.com/PEOPLE-ER/Spectral-Recovery.git#egg=spectral_recovery
```

or from a local clone of the repository:

```bash
git clone https://github.com/PEOPLE-ER/Spectral-Recovery.git
cd Spectral-Recovery
pip install -e .
```