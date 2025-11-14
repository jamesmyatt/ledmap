# 6. Maps are Arrays

Date: 2025-11-14

## Status

Accepted

Updates [#2][ADR-2]

## Context

Implementing the mapping as custom object has been tricky.
In particular:

- keeping all the required data and logic together
- confusion between generators and operations

This was a consequence of [ADR-2][ADR-2].
We need to review whether there is a better way,
potentially using additional libraries.

## Decision

We will rewrite the library using arrays to store the maps.
We will use NumPy to implement the arrays.

## Justification

An array stores the index of the pixel at a given location:
a 1D array represents a string or line of pixels,
and a 2D array represents a matrix of pixels.
Standard array manipulation functions can then be used
to implement the operations associated with creating the map.

NumPy is the best simple array manipulation library.
We have found no functional ones that are simpler,
even if the accelerated routines and advanced calculations
in NumPy are not required for this application.

## Consequences

- Add NumPy as dependency
- Rewrite core logic using arrays

[ADR-2]: ./0002-standard-library-only.md
