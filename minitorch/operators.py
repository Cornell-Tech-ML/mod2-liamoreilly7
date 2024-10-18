"""Collection of the core mathematical operators used throughout the code base."""


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

import math

from typing import Callable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x (float): The first number to multiply.
        y (float): The second number to multiply.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The number to return.

    Returns:
    -------
        float: The number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x (float): The first number to add.
        y (float): The second number to add.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if x is less than y.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if x is equal to y.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if x is close to y.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x is close to y, False otherwise.

    """
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of x.

    Args:
    ----
        x (float): The number to calculate the sigmoid of.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the ReLU of x.

    Args:
    ----
        x (float): The number to calculate the ReLU of.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Calculates the natural logarithm of x.

    Args:
    ----
        x (float): The number to calculate the natural logarithm of.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential of x.

    Args:
    ----
        x (float): The number to calculate the exponential of.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the inverse of x.

    Args:
    ----
        x (float): The number to calculate the inverse of.

    Returns:
    -------
        float: The inverse of x.

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x (float): The input to the log function.
        y (float): Second argument.

    Returns:
    -------
        float: The derivative of log times a second arg.

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of inv times a second arg.

    Args:
    ----
        x (float): The input to the inv function.
        y (float): Second argument.

    Returns:
    -------
        float: The derivative of inv times a second arg.

    """
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu times a second arg.

    Args:
    ----
        x (float): The input to the relu function.
        y (float): Second argument.

    Returns:
    -------
        float: The derivative of relu times a second arg.

    """
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[T], R], x: List[T]) -> List[R]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        f: A function that takes an element of type T and returns type R.
        x: A list of elements of type T.

    Returns:
    -------
        A list of elements of type R, resulting from applying f to each element in x.

    """
    return [f(item) for item in x]


def zipWith(f: Callable[[T, S], R], x: List[T], y: List[S]) -> List[R]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        f: A function that takes two arguments of types T and S and returns type R.
        x: A list of elements of type T.
        y: A list of elements of type S.

    Returns:
    -------
        A list of elements of type R, resulting from applying f to pairs of elements from x and y.

    """
    return [f(x[i], y[i]) for i in range(len(x))]


def reduce(f: Callable[[T, T], T], x: List[T]) -> T:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        f: A function that takes two arguments of type T and returns type T.
        x: A list of elements of type T.

    Returns:
    -------
        The result of applying f cumulatively to the first two items of x.

    """
    if not x:
        raise ValueError("Cannot reduce an empty list")
    result = x[0]
    for item in x[1:]:
        result = f(result, item)
    return result


def negList(x: List[float]) -> List[float]:
    """Negate all elements in a list using map.

    Args:
    ----
        x (List[float]): A list of floating-point numbers.

    Returns:
    -------
        A new list with all elements negated.

    """
    return map(neg, x)


def addLists(x: List[float], y: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        x (List[float]): A list of floating-point numbers.
        y (List[float]): A list of floating-point numbers.

    Returns:
    -------
        A new list with corresponding elements added.

    """
    return zipWith(add, x, y)


def sum(x: List[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        x (List[float]): A list of floating-point numbers.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(add, x)


def prod(x: List[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        x (List[float]): A list of floating-point numbers.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, x)
