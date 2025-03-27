# zeroshot-engine

A open-source scientific zero-shot text classification engine based on various LLM models.

## 📖 About this package

### Description

This project provides a flexible framework for performing zero-shot classification using large language models and pandas. It allows you to classify text into categories without requiring explicit training data for those categories. All instructions to LLMs are provided by mere natural language prompts. The framework is designed to support a wide range of text classification tasks including multi-label, multi-class, and single-class classification scenarios.

### Purpose

This package was developed as part of an academic research project to systematically classify political communication. The primary goal was to create an easy-to-use and accessible framework for building adaptable zero-shot classifications with large language models (LLMs) across a wide variety of text analysis tasks. By providing a flexible and intuitive tool, this project aims to empower students and researchers — especially those in social sciences — to explore, evaluate, and harness the potential of zero-shot classification while addressing its challenges in a user-friendly environment. I have no financial interest in this project.

### Open-Source and Non-Commercial

This project is fully open-source and was developed with no financial interests. It is intended to support academic research and the broader scientific community. Contributions are welcome to help improve the framework and expand its capabilities.

## ✨ Features
### Overview
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

### Core Modules
* **Iterative Double Validated Zero-Shot Classification (IDZSC)**: IDZSC is the core module to classify texts in an iterative process. It can use a double validation technique to ensure the robustness and accuracy of the classifications.
* **Hierarchical Double Validated Zero-Shot Classification (HDZSC)**: HDZSC extends the zero-shot classification capabilities to hierarchical category structures. It leverages a double validation approach to maintain accuracy while navigating the complexities of hierarchical classification.

## 🚀 Get Started
### How to install
Install the `zeroshot-engine` package using pip in your Windows Powershell or Linux / Mac Bash Terminal.
```bash
pip install zeroshot-engine
```

### Interactive Demo in the Command Line
Test the `zeroshot-engine` in the HDZSC-scenario by selecting from a wide variety of LLMs and bringing your own text for classification:
```bash
zeroshot-engine demo
```
This command will guide you through an interactive demo where you can:

* Choose an LLM model (e.g., one from OpenAI or Ollama).
* Provide your own text for classification or use a provided example text.
* Observe how the hierarchical classification process works in real-time.

### Run your first zeroshot project in python

**[Tutorial: Get started with your first classification](docs/Tutorial_Get_Started.md)**: This tutorial provides example code for your first test project, which you can use as a template to build and adapt your own research projects. For more detailed information and advanced usage, please refer to the [[Documentation]].

### Documentation
For more detailed information about the framework and its implementation, please refer to the following documentation:

* **[Overview of IDZSC and HDZSC](docs/Overview_IDZSC_and_HDZSC.md)** - A comprehensive explanation of the Iterative and Hierarchical Double Zero-Shot Classification approaches, including detailed examples and usage patterns.

* **[Performance Evaluation](docs/Performance_Evaluation.md)** - Benchmark results and performance metrics across different models and classification tasks.

* **[In-Depth-Demo-Explanation](docs/HDZSC_Demo_Showcase.md)** - Learn how the HDZSC works in detail.

* **[Tutorial: Get started with your first classification](docs/Tutorial_Get_Started.md)** - Create your first projects with prompt, code examples and text to learn how to set up the classifer.

### Example Flow Chart
```
==============================================================
        ZEROSHOTENGINE DEMO LABEL DEPENDENCY FLOWCHART           
==============================================================

 [POLITICAL]
 ├─ if political = 1:
 │   [PRESENTATION]
 │   [ATTACK]
 │   ├─ if attack = 1:
 │   │   [TARGET]
 │   │   │
 │   │   ▼
 │   │   STOP
 │   └─ if attack = 0:
 │       → Skip: target
 │       STOP
 └─ if political = 0:
     → Skip: presentation, attack, target
     STOP

--------------------------------------------------------------
                 STOP CONDITIONS EXPLANATION                  
--------------------------------------------------------------
  If political = 0 (absent), the following steps are skipped:
    - presentation
    - attack
    - target

  If attack = 0 (absent), the following steps are skipped:
    - target

--------------------------------------------------------------
                            LEGEND                            
--------------------------------------------------------------
 - 1 (present): Proceeds to the next classification step
 - 0 (absent): Skips one or more subsequent classifications

 LABEL CODES 
    present: 1
    absent: 0
    non-coded: 8
    empty-list: []

--------------------------------------------------------------
```

## 📅 Planned Features

*   List of supported LLMs
*   Additional tutorial for double validation vs. zero-temp approach.
*   Documentation of all relevant functions.
*   Create prompting guidelines.
*   Better integration and testing of validation metrics.
*   Automated Logging System
*   Add contribution guidelines.
*   Support for more LLMs and APIs.

## 🚧 Notice: Under Development

> **Note:**  
> While the core functionality of `zeroshot-engine` is already up and running, this project is still under active development.
> There may be bugs, incomplete features, or areas for improvement.  
> 
> If you encounter any issues, have feature requests, or would like to contribute code to the project, please feel free to:  
> - Open an issue on the [GitHub repository](https://github.com/your-repo-link/issues).  
> - Submit a pull request with your contributions.  
> - Contact the author directly at **luc.schwarz@posteo.de**.  
> 
> Contributions are highly appreciated and will help improve the framework for the scientific community!

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🫱🏼‍🫲🏼 Contributing

Contributions are welcome! Feel free to open issues for bug reports or feature requests. If you'd like to contribute code directly, please see the [contributing guidelines](CONTRIBUTING.md).

## 🤵 Author

Lucas Schwarz

## 📧 Contact

luc.schwarz@posteo.de

## 🏛️ Citation

If you use `zeroshot-engine` in your research, please cite it as follows:

> Schwarz, L. (2025) „zeroshot-engine: A scientific zero-shot text classification engine based on various LLM models“. Zenodo. doi: 10.5281/zenodo.15079108.

```bibtex
@software{schwarz_2025_15079108,
  author       = {Schwarz, Lucas},
  title        = {zeroshot-engine: A scientific zero-shot text
                   classification engine based on various LLM models
                  },
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.1.2},
  doi          = {10.5281/zenodo.15079108},
  url          = {https://doi.org/10.5281/zenodo.15079108},
}
```
