"""
Simple replacement for better_abc package.
This module provides abstract_attribute functionality for transformer_lens.
"""


class AbstractAttribute:
    """
    A descriptor that acts as an abstract attribute.
    Subclasses must override this attribute or a NotImplementedError will be raised.
    """
    def __init__(self, doc=None):
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        raise NotImplementedError(
            f"Abstract attribute not implemented in {objtype.__name__}"
        )

    def __set__(self, obj, value):
        # Allow setting the value during initialization
        obj.__dict__[self._name] = value

    def __set_name__(self, owner, name):
        self._name = name


def abstract_attribute(doc=None):
    """
    Create an abstract attribute that must be implemented by subclasses.
    
    Args:
        doc: Optional documentation string for the attribute
        
    Returns:
        AbstractAttribute descriptor
    """
    return AbstractAttribute(doc)
