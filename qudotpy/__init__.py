"""The qudotpy module for the emulation of the gate model of quantum computation
using matrices.
"""
# -*- coding: utf-8 -*-
try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
