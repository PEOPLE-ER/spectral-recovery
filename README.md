# Spectral Recovery Tool 

## Set-up

### Installation and Environment

First, clone the repository

```{bash}
(base) $ git clone https://github.com/PEOPLE-ER/Vegetation-Recovery-Assessment.git

```

Then navigate to the project folder and create/activate your conda environment

```{bash}
(base) $ conda env create -f environment.yml
(base) $ conda activate people 
```

### Running

Run the package with the following command

```{bash}
(people) $ python -m spectral_recovery.spectral_recovery
```

Arguments to the tool can be changed in the `spectral_recovery` module, inside
the `if __name__ == '__main__':` function. The module is currently set-up to be 
run over a simple 1-pixel data set, computes all implemented metrics.

### Tests

Units tests can be run with the following command

```{bash}
(people) $ pytest
```
