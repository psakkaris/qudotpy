QuDotPy
=======

A quantum computing library written in Python. Exploring quantum computing has never been easier. With QuDotPy you can 
experiment with single-qubit operations and gates. You can build multiple-qubit states and perform measurements, and finally you can emulate quantum circuits. Our quantum circuit design makes it easy to test out circuit ideas and to step through a circuit like you would a debugger.

To help you get started we have written a detailed usage tutorial that covers most aspects of QuDotPy. The tutorial can be found here: http://psakkaris.github.io/qudotpy/

QuDotPy depends on Numpy. You will need to have Numpy installed before you can use QuDotPy. 


Getting Started
===============

QuDotPy depends on Python 2.7 and is specifically tested against Pythong 2.7.6

You can test by running the unit tests in the parent qudotpy directory
```
python -m unittest qudotpy.test_qudotpy
```


You can always clone this repository and check out the code. However, if you just want to get a feel for QuDotPy just press the 'Download Zip' button on the right panel. This will download the directory **qudotpy-master**

Next run the python shell from the qudotpy directory

```
$ python
>>> from qudotpy import qudot
>>> print qudot.apply_gate(qudot.H, qudot.ZERO)
(0.707106781187+0j)|0> + (0.707106781187+0j)|1>

>>> 

```

You can also install the library as a package thanks to the contributions of sfinucane. From the parent directory
you can now run:
```
python setup.py install
```
and qudotpy will be available on your system.

That's it! For more check out our tutorial http://psakkaris.github.io/qudotpy/

Giving Back
===========

Feel free to contribute back to QuDotPy. If you are new to GitHub read this: 

https://help.github.com/articles/fork-a-repo

If you find a bug, would like to suggest an enhancement or would like to see enhancements we have planned for the future; check out the Issues page.
