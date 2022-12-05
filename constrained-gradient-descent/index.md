---
categories:
- maths
- python
- data
date: '2021-06-29T07:32:46+10:00'
image: /images/constrained_gradient_descent.png
title: Constrained Gradient Descent
---

Gradient descent is an effective algorithm for finding the local extrema of functions, and the global extrema of convex functions.
It's very useful in machine learning for fitting a model from a family of models by finding the parameters that minimise a loss function.
However sometimes you have extra constraints; I recently worked on a problem where the maximum value of the fitted function had to occur at a certain point.
It's straightforward to adapt gradient descent with differentiable equality constraints.

The idea is simple, we've got a function `loss` that we're trying to maximise subject to some `constraint` function.
With gradient descent we will take a step in the direction of greatest decrease of the loss function, along the gradient.
The size of the step we take is called the learning rate, `lr`.
In Pytorch:

```python
def gradient_descent_step(x, loss, lr=1e-3):
    # Set gradient to zero
    if x.grad is not None:
        x.grad.zero_()

    # Calculate the derivative of the loss function
    loss().backward()

    # Step in direction of greatest local decrease
    with torch.no_grad():
        x -= lr * x.grad
```

However this may take us off of our constraint curve.
As long as we are at a point `x` on the constraint curve, `constraint(x) = 0`, we want to stay on that curve.
That means we want to take a step in the direction where the derivative of the constraint is zero (so the value won't change).
This happens in the direction *orthogonal* to the gradient of the constraint, which can be done by removing the component parallel to the constraints gradient.
In Pytorch:

```python
def gradient_descent_step(x, loss, constraint, lr=1e-3):
    # Set gradient to zero
    if x.grad is not None:
        x.grad.zero_()
        
    # Calculate gradient of the constraint
    constraint(x).backward()
    direction = x.grad.clone()
    
    # Calculate gradient of loss function
    x.grad.zero_()
    aloss = loss()
    aloss.backward()
    
    # Remove the projection of the loss gradient onto the constraint gradient.
    # The resulting vector will be perpendicular to the gradient of the constraint.
    perp_proj = x.grad - (x.grad @ direction) / (direction @ direction) * direction
    
    # Step in this direction
    with torch.no_grad():
        x -= lr * perp_proj
```

We can put this into highfalutin differential geometry terminology.
By the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) the equality constraint defines a submanifold of the overall space (except in pathological regions, but there's often an area where this is true).
We want to optimise the loss function on this submanifold.
This is done by projecting the derivative of the loss function on the manifold to the tangent space of the submanifold defined by the constraints.

In fact the method of [Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) solves this exact problem, however on my problem I had difficulty getting the point back to the constraint curve.
However this method of projection worked really well.
We could potentially use a similar approach for an inequality constraint; by first searching in the interior of the region and applying the gradient projection along the boundary ([Boyd and Vandenberghe's *Convex Optimisation*](https://web.stanford.edu/~boyd/cvxbook/) has the details).