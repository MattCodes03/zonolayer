# Zonolayer

**Matthew McCann**  
_University of Strathclyde, 2025_

Developed with guidance and support from **[Marco de Angelis](https://github.com/marcodeangelis)**, University of Strathclyde

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/zonolayer)](https://pypi.org/project/zonolayer/)

**Zonolayer** is a Python package for **last-layer uncertainty modeling via zonotopic representations**.  
It provides **zonotopic output bounds** and **statistical prediction intervals** for neural networks with interval-bounded outputs, enabling precise and interpretable uncertainty quantification in regression tasks.

---

## Features

- Compute **zonotopic bounds** for last-layer outputs.
- Combine **statistical prediction intervals** with interval uncertainty.
- Compatible with PyTorch networks exposing latent features.
- Modular, research-friendly, and easy to use.

By default, Zonolayer relies on **NumPy** for all numerical computations and interval handling.  

If you require more advanced interval arithmetic (e.g., using [`pyinterval`](https://pypi.org/project/pyinterval/) or other specialized packages), you can install the optional dependencies and modify the code accordingly, or submit a request for support to be added. I appreciate any and all feedback.

---

## Installation

Ensure all requirements from `requirements.txt` are installed to reduce any potential issues.

```bash
pip install zonolayer
```

---
## Getting Started
See `basic_usage.py` in the examples directory for the quickest method to get up and running and start experimenting with the library.

### Example Plots

Zonolayer produces zonotopic bounds and statistical prediction intervals. Example outputs:

![Zonotopic bounds](examples/ExampleOutput.png)
![Zonotopic bounds 2](examples/ExampleOutput2.png)


