========
FinTorch
========


.. image:: https://img.shields.io/pypi/v/fintorch.svg
        :target: https://pypi.python.org/pypi/fintorch


.. image:: https://readthedocs.org/projects/fintorch/badge/?version=latest
        :target: https://fintorch.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://codecov.io/gh/AI4FinTech/FinTorch/graph/badge.svg?token=OBD2MHP5SE
 :target: https://codecov.io/gh/AI4FinTech/FinTorch


AI4FinTech project repository


* Free software: MIT license
* Documentation: https://fintorch.readthedocs.io.




FinTorch - Machine Learning for FinTech
=========================================

The integration of AI in the financial sector demands specialized tools that can handle the unique challenges of this field, especially in regulatory compliance and risk management. Building on the familiarity and robustness of PyTorch, FinTorch aims to bridge the gap between AI technology and the financial industry needs.

Goal
----
Develop FinTorch, an open-source machine learning library as an extension of PyTorch, specifically tailored for the FinTech industry's compliance and risk management requirements.

Key Objectives
--------------

1. Specialized Financial AI Models
   Implement state-of-the-art machine learning models for financial data analysis, fraud detection, risk assessment, and regulatory compliance, seamlessly integrating with PyTorch's existing framework.

2. Regulatory Compliance Toolkit
   Provide tools specifically designed for monitoring and ensuring adherence to financial regulations using AI.

3. User-Friendly API
   Maintain a tensor-centric API, consistent with PyTorch, ensuring ease of use for those familiar with PyTorch. Aim for simplicity, where basic models can be implemented in as few as 10-20 lines of code.

4. Extensibility for Research
   Offer a flexible platform for academic and industry researchers to develop and test new AI models for FinTech, with support for custom architectures and novel strategies.

5. Scalability and Real-World Application
   Focus on scalability to handle large-scale financial data and real-world scenarios.

6. Ethical and Responsible AI Practices
   Embed principles of sustainable and responsible AI, ensuring that models adhere to ethical standards and contribute positively to the FinTech ecosystem.

7. Educational Resources and Community Support
   Provide comprehensive documentation, tutorials, and masterclasses to facilitate learning and collaboration within the AI4FinTech community.

Impact
------
FinTorch will not only streamline the process of regulatory compliance for FinTech companies but also foster innovation and research in AI-driven financial technologies. It will serve as a crucial tool for industry professionals, researchers, and government institutions, aligning with the AI4FinTech community's objectives of knowledge dissemination and development of responsible, cutting-edge financial solutions.


Getting started
---------------
Please install the package as follows

.. code-block:: bash

   pip install fintorch

**Required Dependencies**

The following dependencies must be installed:

.. code-block:: bash

   pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

**Important Notes**

* Replace `${TORCH}` and `${CUDA}` with the appropriate version numbers for your environment (e.g., "1.12.0" and "cu113").
* These installation commands use custom index URLs provided by PyTorch Geometric (PyG).



Description of the Structure
-----------------------------

- `fintorch` Directory: Contains the core library modules.
    - `models`: Core models for compliance monitoring, fraud detection, risk assessment, and sustainable finance.
    - `datasets`: Financial datasets and data processing utilities.
    - `utils`: Helper tools and functions for compliance and other financial applications.
    - `training`: Training and evaluation scripts for the models.

- `examples` Directory: Example scripts demonstrating the use of FinTorch in different scenarios.
- `tests` Directory: Unit and integration tests for the library.
- `benchmarks` Directory: Benchmark scripts and resources for testing the performance of the library.
- `docs` Directory: Documentation files, including build scripts and source files.
- `docker` Directory: Dockerfile and related resources for containerizing the FinTorch library.
- `conda` Directory: Scripts and files needed for building a Conda package of the library.
- `tutorials` Directory: Jupyter notebooks that provide tutorials on how to use the library for various FinTech applications.
