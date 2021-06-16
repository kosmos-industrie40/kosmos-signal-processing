========================
kosmos-signal-processing
========================

This project aims to offer helpful functions for especially signal or time series exploration
and processing.

It offers:

* an evaluation algorithm designed to find the best interpolation method
  based on your data set
* an FFT implementation that serves as an inverse bandpass filter and
  filters out a dominant frequency in order to further focus on outliers


Installation
============
Installation from source
++++++++++++++++++++++++

.. code-block:: bash

    # Clone this project then...
    cd kosmos-signal-processing
    pip install .

Development setup
+++++++++++++++++

.. code-block:: bash

    # Clone this project then...
    cd kosmos-signal-processing
    # Create a virtualenv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    # To run tests
    python setup.py test

How to use
==========

After you installed the package

1. Using the interpolation evaluation method

.. code-block:: python

    import pandas as pd
    import numpy as np
    from kosmos_signal_processing.interpolate import interpol_method_selection

    data: pd.DataFrame = pd.DataFrame(np.random.randn(100, 1), columns=["y"])

    best_method: str
    score: int
    best_method, score = interpol_method_selection(data)
    print("The best interpolation method for this random data is {} with a score of {}".format(best_method, score))

2. Using the inverse bandpass filter

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from kosmos_signal_processing.fourier import filter_signal

    signal_x: np.ndarray = np.linspace(0, 2 * np.pi, 1000)
    signal_y: np.ndarray = (
            np.sin(10 * signal_x)
            + np.sin(50 * signal_x)
            + np.sin(60 * signal_x)
            + np.sin(100 * signal_x)
            + 2
    )
    # Plot raw signal
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,5))
    ax1.plot(signal_x, signal_y)
    # Filter signal
    filtered_signal: np.ndarray = filter_signal(signal=signal_y)
    # Plot filtered signal
    ax2.plot(signal_x, filtered_signal, color="orange")
    plt.show()



Contributing and Outlook
========================

Contributions to this project are highly encouraged since there are several further useful functions
which may help the developer to better explore a signal or time series and/or to transform it
for further data handling.

Here is a list of possible extensions:

#. Extend the interpolation
    * in order to account for multivariate methods
    * in order to account for further not included methods
    * be able to explicitly test only specific methods using arguments
#. Include extrapolation
    * for instance Gaussian Process, which can even be used both for intra- & extrapolation
#. Include further signal processing techniques
    * e.g. wavelet-transformation

License
=======

MIT - see LICENSE.txt

Credit
======

This repository was created during the BMBF research project `KOSMoS <https://www.kosmos-bmbf.de/>`_.
Thanks to the `german ministry of education and research <https://www.bmbf.de/en/index.html>`_!

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
