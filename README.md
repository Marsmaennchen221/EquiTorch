# EquiTorch Software Practical 

Our general approach is to work on a list of neural network architectures with increasing complexity:

1. MLP
2. CNN
3. RNN
4. LSTM
5. Transformer
6. VAE
7. GAN
8. Neural ODE

This order may change as we are unsure of the difficulty/complexity of LSTMs compared to VAEs or Transformers.

## Task 1 (MLP):
Do for Pytorch (Alex) and Equinox (Jens):

- select sample dataset: (Artificial Moons for now)
- verify architecture code, since it's just a one-shotted language model implementation for now.
- implement training loop
- experiment with a tree-sitter to generate syntax tree of:
    - model definition
    - training loop
- identify core concepts of jax that might cause issues on transpilation

## Task 2 (MLP):
- Compare tree-sitter results. 
- Identify:
    - easily translatable building blocks/modules
    - difficult building blocks that have no direct equivalent

## Task 3 (MLP):
Implement a first transpiler version.