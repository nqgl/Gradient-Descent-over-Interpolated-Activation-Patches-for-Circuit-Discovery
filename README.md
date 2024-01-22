# Gradient-Descent-over-Interpolated-Activation-Patches-for-Circuit-Discovery

for gpt-2-small there are $\sum{l=0}{n - 1} l h ^ 2 = n (n + 1) / 2 * h ^ 2 $

$$
\sum
$$

(src-layer, src-head) -> (dest-layer, dest-head) 

The naive implementation of this requires O(n^2 h^2) forward passes if I recall correctly. However you can do parallel src-heads within the same layer together by stacking them into a batch dimension, which brings it down to O(n^2 h) forward passes. My plan was to implement this to get rid of some confusion, then have it require ~200gb of ram and require pairing down the graph to check, which I would do by starting with edges defined as (src-layer, src-head) -> (dest-layer) 
