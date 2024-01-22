# Gradient-Descent-over-Interpolated-Activation-Patches-for-Circuit-Discovery

for gpt-2-small there are 
$h \sum\limits_{l=0}^{n - 1} l h = n (n - 1) / 2 * h ^ 2 = 9504$
 "edges" between attention heads which could be patched. This assigns a learnable coefficient to each of them (as well connections from the corrupted residual stream to the attention heads and from the head to the output), and then does gradient descent over all of them as a method for circuit discovery that I'm exploring.

The psuedocode (glossing over the output coefficients and residual input coefficient) for what is calculated is like:
activations from normal vs corrupted run: 
W : edge weights
forward : normal model forward to get cached values for heads
forward_patched : a patched forward pass with $head_l_h$ values inside the curly braces patched into head $h$ of layer $l$ and values in brackets corresponding to the value of the head to extract

clean <- forward(input) 
dirty <- forward(corrupted input)
for i $\in {0, 1, 2, ..., 12}:
  for j $\in {0, 1, 2, ..., 12}:
    patched_i_j = forward_patched(input)\{
      for l $\in {0, 1, 2, ..., i - 1}
      for h $\in {0, 1, 2, ..., 12}
      head_l_h = clean_i_j * (1 - W_i_j_l_h) + patched_l_h * W_i_j_l_h
    )\[i, j\]

out = forward_patched(input){
  for l $\in {0, 1, 2, ..., 12}:
    for h $\in {0, 1, 2, ..., 12}:
      head_l_h = patched_l_h
}

$$
O\left(
  h \sum\limits_{l=0}^{n - 1} l h 
\right)
=O\left(
n^2 h^2 
\right)
$$

$$
\sum
$$

(src-layer, src-head) -> (dest-layer, dest-head) 

The naive implementation of this requires O(n^2 h^2) forward passes if I recall correctly. However you can do parallel src-heads within the same layer together by stacking them into a batch dimension, which brings it down to O(n^2 h) forward passes. This is pretty absurd and implausibly workable. Nevertheless, it's what I wanted to work on. 

My plan was to start by implementing this mostly to clarify some things for myself, then have it require an utterly unworkable amount of vram, after which I would start by pairing down the graph prior to doing edge-level processing. I still think this is a good idea, as running a single iteration takes 10-30 seconds and at a batch size of 8 it uses ~15gb of VRAM

Nevertheless, it works a little bit? On not very many iterations so I have to turn way up the  
