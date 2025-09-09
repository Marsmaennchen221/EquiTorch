## Automated Differentiation in Pytorch vs Equinox

Both Frameworks rely on the concept of Reverse-Mode Automatic Differentiation, the main difference is that pytorch does it on the basis of runtime information, building a computation graph along the
execution of a forward pass. In case the forward pass contains e.g. conditional statements, pytorch can just follow the execution path the curent runtime environment did follow.

In contrast to that JAX performs differentiation 'just-in-time'. This means that the JAX transformation system performs a 'trace' of the to-differentiate function at the first time this function is be
called with a specific signature, and build a computation graph on the basis of the runtime information that is available at that moment. The computation graph itself is now a static artifact that
will be reused on further calls.

To transpile pytorch based functions, can make it necessary to make conditional statements in the function dependend on a static argument or to rewrite parts of the control flow with jax.lax
primitives. (TODO check that out in greater detail.)