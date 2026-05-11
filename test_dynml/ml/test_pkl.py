"""Test the ``dynml.ml.pkl`` module.

This module tests the ``dynml.ml.pkl`` module.
"""


# import built-in python-package code
from os import remove
# import external python-package code
# None
# import internal python-package code
from dynml.ml.pkl import load, save
from dynml.utils.config import config


# test the both the load and save methods
class ClassExample():
    """Represent an example class.

    This class represents an example class.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``attribute`` (``int``): an attribute

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the class with ``self.attribute`` equal to 0

    | **References**
    |   None
    """

    def __init__(self) -> None:
        """Initialize the class with ``self.attribute`` equal to 0.

        This method initializes the class with ``self.attribute`` equal to 0.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the attribute with a value of 0
        self.attribute = 0


def test_load_save() -> None:
    """Test the ``load()`` and ``save()`` methods.

    This method tests the ``load()`` and ``save()`` methods.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # configure the test
    config(64, 0)
    # test code for load()
    # create an instance of ClassExample
    test = ClassExample()
    # test the attribute
    assert test.attribute == 0
    # test save()
    save(test, "./test.pkl")
    # load the instance
    loaded = load("./test.pkl")
    # test the attribute
    assert loaded.attribute == 0
    # remove the file
    remove("./test.pkl")
