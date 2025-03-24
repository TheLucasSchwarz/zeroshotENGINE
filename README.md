# zeroshot-engine

A scientific zero-shot text classification engine based on various LLM models.

## Description

This project provides a flexible framework for performing zero-shot classification using large language models and pandas. It allows you to classify text into categories without requiring explicit training data for those categories. All instructions to LLMs are provided by mere natural language prompts. The framework is designed to support a wide range of text classification tasks including multi-label, multi-class, and single-class classification scenarios.

## Features

*   Handles multi-label, multi-class, and single-class classification tasks.
*   Option for incorporating few-shot learning through the flexible prompt engineering approach.
*   Supports multiple LLM models (e.g., OpenAI, Ollama).
*   Easy-to-use command-line interface for demo purposes.
*   Customizable prompts.
*   Integration with pandas for data handling.

### Key Concepts

*   **Zero-Shot Learning:** The ability of a model to make predictions on unseen classes or tasks without prior training on those specific classes or tasks. The system learns entirely through natural language instructions, eliminating the need for labeled examples or fine-tuning.
*   **Sequential Classification:** A process where tasks are performed in a series of steps without strict dependencies (IDZSC approach).
*   **Hierarchical Classification:** A structured approach that breaks down complex classification tasks into a series of simpler decisions following a predefined hierarchy with explicit dependencies (HDZSC approach).
*   **Multi-Prompting:** The use of multiple different prompts for different tasks to elicit more comprehensive and reliable predictions from the model.
*   **Modular Prompt Design:** While not automated in the current implementation, the modular prompt design with text blocks facilitates manual testing and refinement of prompts to improve classification accuracy.


## Installation

```bash
pip install zeroshot-engine
```

## Demo

```bash
zeroshot-engine demo
```

## Usage

```bash
zeroshot-engine --help
```

## Core Modules

### Iterative Double Validated Zero-Shot Classification (IDZSC)

IDZSC is a core module that refines zero-shot classification results through an iterative process. It uses a double validation technique to ensure the robustness and accuracy of the classifications.

### Hierarchical Double Validated Zero-Shot Classification (HDZSC)

HDZSC extends the zero-shot classification capabilities to hierarchical category structures. It leverages a double validation approach to maintain accuracy while navigating the complexities of hierarchical classification.

### Planned Features

*   Improved documentation and examples.
*   Create prompting guidelines.
*   Better integration and testing of validation metrics.
*   For structured benchmarking and prompt engineering approach.
*   Automated Logging System
*   Add contribution guidelines.
*   Support for more LLMs and APIs.

### Documentation
For more detailed information about the framework and its implementation, please refer to the following documentation:

* [Overview of IDZSC and HDZSC](docs/Overview_IDZSC_and_HDZSC.md) - A comprehensive explanation of the Iterative and Hierarchical Double Zero-Shot Classification approaches, including detailed examples and usage patterns.

* [Performance Evaluation](docs/Performance_Evaluation.md) - Benchmark results and performance metrics across different models and classification tasks.


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Feel free to open issues for bug reports or feature requests. If you'd like to contribute code directly, please see our [contributing guidelines](CONTRIBUTING.md).

## Author

Lucas Schwarz

## Contact

luc.schwarz@posteo.de

## Citation

If you use `zeroshot-engine` in your research, please cite it as follows:

> Schwarz, L. (2025) „zeroshot-engine: A scientific zero-shot text classification engine based on various LLM models“. Zenodo. doi: 10.5281/zenodo.15077242.

```bibtex
@software{schwarz_2025_15077242,
  author       = {Schwarz, Lucas},
  title        = {zeroshot-engine: A scientific zero-shot text
                   classification engine based on various LLM models
                  },
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.1.1},
  doi          = {10.5281/zenodo.15077242},
  url          = {https://doi.org/10.5281/zenodo.15077242},
}
```


![PyPI Publishing](https://github.com/TheLucasSchwarz/zeroshot-engine/actions/workflows/python-publish.yml/badge.svg)
