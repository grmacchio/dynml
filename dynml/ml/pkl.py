"""Define a saving and loading method for objects saved using ``pickle``.

This module defines a saving and loading method for objects saved using
``pickle``.
"""


# import built-in python-package code
from os import makedirs
from os.path import dirname
from pickle import dump, HIGHEST_PROTOCOL
from pickle import load as pickle_load
from typing import Any
# import external python-package code
# None
# import internal python-package code
# None


# export public code
__all__ = ["save",

           "load"]


# define the saving function
def save(obj: Any, filename: str) -> None:
    """Save an object using ``pickle``.

    This method saves an object using ``pickle``.

    | **Args**
    |   ``obj`` (``Any``): the object to save
    |   ``filename`` (``str``): the name of the file

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # create the directory if it does not exist
    makedirs(dirname(filename), exist_ok=True)
    # save self to the file location
    with open(filename, "wb") as file:
        dump(obj, file, HIGHEST_PROTOCOL)


# define the loading function
def load(filename: str) -> Any:
    """Load an object that was saved using ``pickle`` from a file.

    This method loads an object that was saved using ``pickle`` from a file.

    | **Args**
    |   ``filename`` (``str``): the name of the file

    | **Return**
    |   ``Any``: the pickle-saved object that was saved to the file

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # load the object from the file
    with open(filename, "rb") as file:
        return pickle_load(file)
