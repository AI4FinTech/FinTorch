.. _fintorch_cli:

FinTorch CLI Interface
======================

The FinTorch Command Line Interface (CLI) provides a convenient way to interact with the FinTorch package. It allows you to download financial datasets, train financial models using hyperparameter sweeps, and utilize additional functionalities.

fintorch: The Main Command
---------------------------

The ``fintorch`` command serves as the entry point for accessing all the subcommands provided by the CLI.

.. code-block:: bash

    fintorch <command> [options]

**fintorch datasets**

This command enables you to download various financial datasets for your analysis.

.. code-block:: bash

    fintorch datasets <dataset>

*   **dataset:** Specifies the name of the dataset you want to download (e.g., "elliptic", "ellipticpp").

**Example:**

.. code-block:: bash

    fintorch datasets elliptic

This will download the "elliptic" dataset to the specified location (`~/.fintorch_data` by default).

**fintorch sweep**

The ``sweep`` command allows you to perform hyperparameter sweeps for your financial models. It leverages the Optuna library to optimize model performance by exploring different combinations of hyperparameters.

.. code-block:: bash

    fintorch sweep --model <model> --predict <predict> [--max_epochs <max_epochs>]

*   **--model:** (Required) Specifies the name of the model to use (e.g., "graphbean_elliptic").
*   **--predict:** (Required) Specifies the type of prediction to perform (e.g., "link_prediction", "node_classification").
*   **--max_epochs:** (Optional) Sets the maximum number of epochs for model training during the sweep.

**Example:**

.. code-block:: bash

    fintorch sweep --model graphbean_elliptic --predict link_prediction --max_epochs 10

This will initiate a hyperparameter sweep for the "graphbean_elliptic" model, optimizing for link prediction with a maximum of 10 training epochs.

fintrainer (Additional Command)
-------------------------------

The ``fintrainer`` command is a versatile tool that directly links to the ``lightningcli``. It allows you to run specific configurations of a model using a configuration file.

.. code-block:: bash

    fintrainer --config <config_file>

*   **--config:** (Required) Specifies the path to the configuration file (e.g., "fintorch/models/graph/graphbean/GraphBEANModule.yaml").

**Example:**

.. code-block:: bash

    fintrainer --config fintorch/models/graph/graphbean/GraphBEANModule.yaml

This will execute the graphbean model using the settings defined in the specified configuration file.

**Note:** The ``fintrainer`` command is not explicitly defined, it is a wrapper around ``lightningcli``. Thus, all the ``lightningcli`` options are available.
