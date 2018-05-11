![CircleCI](https://circleci.com/gh/psakkaris/qudotpy.svg?style=svg)

QuDotPy
=======

A quantum computing library written in Python. Exploring quantum computing has never been easier. With QuDotPy you can 
experiment with single-qubit operations and gates. You can build multiple-qubit states and perform measurements, and finally you can emulate quantum circuits.

To help you get started we have written a detailed usage tutorial that covers most aspects of QuDotPy. The tutorial can be found here: <a href="http://www.sakkaris.com/tutorials/qudotpy.html" target="_blank">QuDotPy Tutorial</a>

QuDotPy depends on Numpy. You will need to have Numpy installed before you can use QuDotPy. 


Getting Started
===============

QuDotPy depends on Python 3 and is specifically tested against Pythong 3.6.5

You can test by running the unit tests in the parent qudotpy directory
```
python -m unittest qudotpy.test_qudotpy
```

```
$ python
>>> from qudotpy import qudot
>>> print qudot.apply_gate(qudot.H, qudot.ZERO)
(0.707106781187+0j)|0> + (0.707106781187+0j)|1>

>>> 

```

That's it! For more check out our tutorial: <a href="http://www.sakkaris.com/tutorials/qudotpy.html" target="_blank">QuDotPy Tutorial</a>
