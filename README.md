# zeroshot-engine

A open-source scientific zero-shot text classification Python package based on various LLM models.

## 📖 About this package

### Description

This project provides a flexible framework for performing zero-shot classification using large language models in Python. It allows you to classify text into categories without requiring explicit training data for those categories. All instructions to LLMs are provided by mere natural language prompts. The framework is designed to support a wide range of text classification tasks including multi-label, multi-class, and single-class classification scenarios.

### Purpose

This package was developed as part of an academic research project to systematically classify political communication. The primary goal was to create an easy-to-use and accessible framework for building adaptable zero-shot classifications with large language models (LLMs) across a wide variety of text analysis tasks. By providing a flexible and intuitive tool, this project aims to empower students and researchers — especially those in social sciences — to explore, evaluate, and harness the potential of zero-shot classification while addressing its challenges in a user-friendly environment. I have no financial interest in this project.

### Open-Source and Non-Commercial

This project is open-source and was developed with no financial interests. It is intended to support academic research and the broader scientific community. Contributions are welcome to help improve the framework and expand its capabilities.

## ✨ Features
### Overview
*   Flexible **prompt-driven zero-shot classification** (with few-shot option).
*   Supports diverse classification tasks: **multi-label/class, sequential (IZSC), and hierarchical (HZSC)**.
*   **Multi-API Support**: Seamlessly switch between APIs like OpenAI, OpenRouter, and local models via Ollama or connect to any OpenAI-compatible API (e.g. Google Gemini API) using custom base URLs.
*   **Parallel Processing**: Classify large pandas DataFrames in parallel to speed up your workflow.
*   **Prompt Templating**: Easily build and manage complex prompts from a structured format.
*   **Validation and Retries**: Improve reliability with built-in validation for dual-prediction setups and automatic retries.
*   Option to perform **double shots**: Perform each step twice and evaluate the performance and uncertainty afterwards.
*   Integrates with multiple LLM providers (e.g., **OpenAI**, **Ollama**, **OpenRouter**) and **pandas DataFrames**.
*   Provides an easy-to-use **command-line interface (CLI)** for demonstration purpose.
*   **Open-source** and non-commercial, designed for research and academic use.

### Key Concepts

*   **Zero-Shot Classification (via Prompting)**: The engine performs *Zero-Shot Classification* by instructing Large Language Models (LLMs) using user-defined natural language prompts. This allows classifying text into new categories without needing specific training examples for those categories; the LLM understands the task solely from the prompt provided through the engine.
*   **Sequential Classification (IZSC Approach):** The *Iterative Zero-Shot Classification (IZSC)* module enables *Sequential Classification*. Within this package, it means users can define a series of potentially independent classification steps executed in order. This allows building flexible, multi-stage analysis pipelines where each stage uses potentially different prompts or focuses on different aspects of the text, without requiring strict hierarchical dependencies between stages.
*   **Hierarchical Classification (HZSC Approach):** The *Hierarchical Zero-Shot Classification (HZSC)* module implements *Hierarchical Classification*. This allows users to structure classification tasks as a tree or hierarchy defined in the configuration. The package guides the LLM through this predefined structure, making classification decisions at one level contingent upon the outcomes of decisions at parent levels.
*   **Multi-Prompting:** The framework supports a *Multi-Prompting* strategy by allowing users to define distinct prompts for each classification step (in IZSC) or for different levels/branches within a hierarchy (in HZSC). This flexibility enables tailoring LLM instructions specifically to the sub-task at hand, aiming for improved accuracy and nuance.
*   **Modular Prompt Design:** The package encourages *Modular Prompt Design*. Prompts can be constructed from reusable text blocks managed within the configuration. While not automated, this structure simplifies the user's process of experimenting with, testing, and refining different prompt components manually to optimize LLM performance.
### Core Modules
* **Iterative Double Validated Zero-Shot Classification (IZSC)**: IZSC is the core module to classify texts in an iterative process. It can use a double validation technique to ensure the robustness and accuracy of the classifications.
* **Hierarchical Double Validated Zero-Shot Classification (HZSC)**: HZSC extends the zero-shot classification capabilities to hierarchical category structures. It leverages a double validation approach to maintain accuracy while navigating the complexities of hierarchical classification.

## 🚀 Get Started
### How to install
Install the `zeroshot-engine` package using pip in your Windows Powershell or Linux / Mac Bash Terminal.
```bash
pip install zeroshot-engine
```

### Interactive Demo in the Command Line
Test the `zeroshot-engine` in the HDZSC-scenario by selecting from a wide variety of LLMs and bringing your own text for classification:

**On Linux/Mac:**
```bash
zeroshot-engine demo
```
**On Windows:**
```bash
zeroshot-engine.exe demo
```
This command will guide you through an interactive demo where you can:

* Choose an LLM model (e.g., one from OpenAI (with Routing Options to e.g. OpenRouter via Base-URL) or Ollama).
* Provide your own text for classification or use a provided example text.
* Observe how the hierarchical classification process works in real-time.

### Run your first Zeroshot Classification Project in Python

This **[tutorial](docs/Tutorial_Get_Started.md)** provides example code for your first test project, which you can use as a template to build and adapt your own research projects. For more detailed information and advanced usage, please refer to the documentation.

## 📚 Documentation
For more detailed information about the framework and its implementation, please refer to the following documentation:

* **[Full Documentation](docs/zeroshot_engine_full_documentation.md)** - The complete documentation for the `zeroshot-engine` package, including detailed explanations of all functions, parameters, and core concepts.

* **[Overview of IZSC and HZSC](docs/Overview_IZSC_and_HZSC.md)** - A comprehensive explanation of the Iterative and Hierarchical Double Zero-Shot Classification approaches, including detailed examples and usage patterns.

* **[Performance Evaluation](docs/Performance_Evaluation.md)** - Benchmark results and performance metrics across different models and classification tasks.

* **[Prompting Explanation](docs/HZSC_Demo_Showcase.md)** - Learn how create the prompt table for the HZSC workflow.

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
*   Create prompting guidelines.
*   Better integration and testing of validation metrics.
*   Automated Logging System
*   Add contribution guidelines.
*   Support for more LLMs and APIs.
*   Agent-based supervising behaviour in the double validation process, that a second series prompt could evaluate the first prompt or resolves conflicts in two prompts, or a thrid prompt with different model could evaluate the conflict if needed between the first identical prompts.

## 🚧 Notice: Under Development  
While the core functionality of `zeroshot-engine` is already up and running, this project is still under active development.
There may be bugs, incomplete features, or areas for improvement.  

If you encounter any issues, have feature requests, or would like to contribute code to the project, please feel free to:  
* Open an issue on the [GitHub repository](https://github.com/TheLucasSchwarz/zeroshotENGINE/issues).  
* Submit a pull request with your contributions.  
* Contact the author directly at **luc.schwarz@posteo.de**.  

Contributions are highly appreciated and will help improve the framework for the scientific community!

##  Model Access and Orchestration
Authorship and provision of Large Language Models (LLMs) reside entirely with their respective creators. zeroshot-engine claims no authorship over these models. It functions solely as an orchestrator, providing a structured framework to access models from providers like OpenAI (using the official openai Python library) or locally hosted Ollama instances (via the official LangChain Python library).

Note: Utilizing Ollama requires it to be installed and running separately on your system. Internal data handling relies on pandas DataFrames, making it a core dependency (familiar if you work with Python for data analysis).

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

> Schwarz, L. (2025). zeroshot-engine: A scientific zero-shot text classification engine based on various LLM models (Version 0.2.0) [Computer software]. [https://doi.org/10.5281/zenodo.16890143](https://doi.org/10.5281/zenodo.17623667)

```bibtex
@software{Schwarz_zeroshot-engine_A_scientific_2025,
        author = {Schwarz, Lucas},
        doi = {10.5281/zenodo.17623667},
        month = mar,
        title = {{zeroshot-engine: A scientific zero-shot text classification engine based on various LLM models}},
        url = {https://github.com/TheLucasSchwarz/zeroshotENGINE},
        version = {0.2.0},
        year = {2025}
}
```
