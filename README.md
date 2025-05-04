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

Im not quite sure about the order, since i don't like LSTMs and think they are actually more difficult/complex than VAEs or Transformers, but we will see.

## Task 1 (MLP):
Do for Pytorch (Alex) and Equinox (Jens):

- select sample dataset
- verify architecture code, since it's just a one-shotted language model implementation for now.
- implement training loop
- experiment with a tree-sitter to generate syntax tree of:
    - model definition
    - training loop

## Task 2 (MLP):
- Compare tree-sitter results. 
- Identify:
    - easily translatable building blocks/modules
    - difficult building blocks that have no direct equivalent

## Task 3 (MLP):
Implement a first transpiler version.