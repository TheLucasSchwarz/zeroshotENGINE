# zeroshot-engine

A scientific zero-shot text classification engine based on various LLM models.

## Description

This project provides a flexible framework for performing zero-shot classification using large language models and pandas. It allows you to classify text into categories without requiring explicit training data for those categories. All instructions to LLMs are provided by mere natural language prompts. 

## Features

*   Supports multiple LLM models (e.g., OpenAI, Ollama).
*   Easy-to-use command-line interface for demo purposes.
*   Customizable prompts.
*   Integration with pandas for data handling.

## Installation

```bash
pip install zeroshot-engine
```

## Usage

```bash
zeroshot-engine --help
```

## Example

```python
from zeroshot_engine import ZeroShotClassifier

classifier = ZeroShotClassifier(model="openai")
categories = ["positive", "negative", "neutral"]
text = "This is a great movie!"
result = classifier.classify(text, categories)
print(result)
```

## Core Modules

### Iterative Double Validated Zero-Shot Classification (IDZSC)

IDZSC is a core module that refines zero-shot classification results through an iterative process. It uses a double validation technique to ensure the robustness and accuracy of the classifications.

### Hierarchical Double Validated Zero-Shot Classification (HDZSC)

HDZSC extends the zero-shot classification capabilities to hierarchical category structures. It leverages a double validation approach to maintain accuracy while navigating the complexities of hierarchical classification.

## Planned Features

*   Improved documentation and examples.
*   Create prompting guidelines.
*   Better integration and testing of validation metrics.
*   Automated Logging System
*   Add contribution guidelines.
*   Support for more LLMs and APIs.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Feel free to open issues for bug reports or feature requests. If you'd like to contribute code directly, please see our [contributing guidelines](CONTRIBUTING.md).

## Author

Lucas Schwarz

## Contact

luc.swz97@gmail.com

## Citation

If you use `zeroshot-engine` in your research, please cite it as follows:

```bibtex
@misc{zeroshotengine,
  author = {Lucas Schwarz},
  title = {zeroshot-engine: A scientific zero-shot text classification engine based on various LLM models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/zeroshot-engine}}
}
```