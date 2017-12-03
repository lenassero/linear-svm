#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" Some tools.
"""

__version__ = "0.1"
__author__ = "Nasser Benabderrazik"

def give_available_values(ls):
    """ Give the available values in ls as: "value1 or value2 or value3 ...".
    This is used for printing the available values for string variables in a 
    function.

    Parameters
    ----------

    ls: list (String)
        List 

    Returns
    -------

    s: String
    """
    n = len(ls)
    if n > 1: 
        s = "'{}' or '{}'" + " or '{}'"*(n-2)
        s = s.format(*[value for value in ls])
    else:
        s = ls[0]
    return s

if __name__ == '__main__':
    print(give_available_values(["newtonLS", "dampedNewton"]))