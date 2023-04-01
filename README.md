# Explainable Sexism

This repository contains the associated code for the Argument Mining class WiSe22-23. 

We go over the multiclass classification of Online Sexism based on the data provided the [SemEval 2023 Task 10B](https://github.com/rewire-online/edos).
A link to the  task paper can be found [here](https://arxiv.org/abs/2303.04222). 

## Setup 

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> NOTE: the code has been tested on Python 3.8.10 

## Data

The data used for training and testing the code can be found at the SemEval Task linked above.

## Usage

There are two main modules in this project: Master of Class Experts (MoCE) and Master of Domain Experts (MoDE).

To run the MoCE module:

```
python moce_main.py
```

To run the MoDE module:

```
python mode_main.py
```
