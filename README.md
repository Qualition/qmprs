# QMPRS
`qmprs` is a state-of-the-art package for approximately compiling quantum circuits from high-level primitives such as statevectors and operators using Quantum Matrix Product Reduced Synthesis (QMPRS). This package enables optimal state-preparation and unitary synthesis using a high fidelity approximation through Matrix Product States (MPS) and Matrix Product Operators (MPO), which enable an exponential reduction from the conventional $O(2^N)$ to $O(N)$ for $N$ qubit operations.

## Getting Started

### Prerequisites

- python 3.11.9

### Quick Installation

`qmprs` can be installed with the command:

```
pip install qmprs
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [`documentation`]().

## Usage

The docs/examples are a good way for understanding how `qmprs` works. Depending on your preference, you may use the package as end-to-end, or use it in parts for low-level modifications.

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