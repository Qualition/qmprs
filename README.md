# QMPRS

<!-- [![PyPI version](https://img.shields.io/pypi/v/qmprs)](//pypi.org/project/qmprs) -->
[![License](https://img.shields.io/github/license/Qualition/qmprs.svg?)](https://opensource.org/licenses/Apache-2.0) <!--- long-description-skip-begin -->
<!-- [![Tests](https://github.com/Qualition/qmprs/actions/workflows/tests.yml/badge.svg)](https://github.com/qualition/qmprs/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/Qualition/qmprs/branch/main/graph/badge.svg?token=IHWJZG8VJT)](https://codecov.io/github/Qualition/qmprs) -->
<!-- [![Codacy Badge](https://app.codacy.com/project/badge/Grade/e287a2eed9e24d5e9d4a3ffe911ce6a5)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) -->


`qmprs` is a state-of-the-art package for approximately compiling quantum circuits from high-level primitives such as statevectors and operators using Quantum Matrix Product Reduced Synthesis (QMPRS). This package enables optimal state-preparation and unitary synthesis using a high fidelity approximation through Matrix Product States (MPS) and Matrix Product Operators (MPO), which enable an exponential reduction from the conventional $O(2^N)$ to $O(N)$ for $N$ qubit operations.

## Getting Started

### Prerequisites

- python 3.10, 3.11, 3.12
- Ubuntu

Currently, due to this [issue](https://github.com/Qualition/quick/issues/11) `qmprs` works reliably only on Ubuntu.

### Quick Installation

`qmprs` can be installed with the command:

```
pip install qmprs
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source:

```
pip install git+https://github.com/Qualition/qmprs
```

## Usage

The notebooks are a good way for understanding how `qmprs` works. Depending on your preference, you may use the package as end-to-end, or use it in parts for low-level modifications.

### Quick Example

```py
import numpy as np
from quick.circuit import QiskitCircuit
from qmprs.synthesis.mps_encoding import Sequential

num_qubits = 8

# Generate a random state
random_state = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
random_state /= np.linalg.norm(random_state)

# Initialize the encoder with the preferred circuit backend
encoder = Sequential(QiskitCircuit)

# Encode the MPS
encoded_circuit = encoder.prepare_state(random_state, num_layers=32, bond_dimension=8)
```

## Testing

Run all tests with the command:

```
py -m pytest tests
```

## Contribution Guidelines

If you'd like to contribute to `qmprs`, please take a look at our [`contribution guidelines`](). By participating, you are expected to uphold our code of conduct.

We use [`GitHub issues`](https://github.com/Qualition/QMPRS/issues) for tracking requests and bugs.

## License

See [`LICENSE`](LICENSE) for details.

## Citation

If you wish to attribute/distribute our work, please cite the accompanying paper:
```
@article{malekaninezhad2024qmprs,
   title={qmprs: {A}pproximate {Q}uantum {C}ircuit {C}ompiler},
   author={Amir Ali Malekani Nezhad, Tushar Pandey},
   year={2024},
   journal={arXiv preprint arXiv:TBD},
}
```