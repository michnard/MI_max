# Are Noise correlations truly information-limiting?
TL;DR: try by yourself! https://michnard.github.io/MI_max

Neural co-variability can be partitioned into an explainable component, driven by external stimuli, oscillations, and other global signals; and an unexplained component, or “noise” correlations, which can lead to profound coding (dis)advantages.
Traditionally, noise correlations are analyzed by comparing the information content of data population responses with an alternative synthetic code where the neurons **preserve their tuning but are otherwise independent**. 
These analyses usually conclude that noise correlations are information-limiting. **Such manipulations do not represent a neural code that is biologically realizable**: in the brain, tuning is affected both by upstream inputs and local network interactions. 

Instead, we conduct our analyses within a stochastic model of spiking neurons ($x_1, x_2 \in \\{ 0 , 1 \\}^2$) with controllable stimulus-dependent upstream inputs $f_1(s),f_2(s)$ and tunable noise correlations $c$, which represent effective local network interactions and unaccounted-for external driving variables.

$$P(x | s) \propto \exp (f_1(s)x_1 + f_2(s)x_2 + c x_1x_2)$$

Try the effects of adding positive, negative, or zero noise correlations to a pair of cells with completely tunable inputs by yourself!

Simply open https://michnard.github.io/MI_max and run the jupyter notebooks.

The same effects carry over to larger and more complicated populations of neurons. If you want to read more about this, please visit: https://www.jneurosci.org/content/early/2023/09/26/JNEUROSCI.0194-23.2023 (or freely accessible, earlier preprint at https://www.biorxiv.org/content/10.1101/2021.09.28.460602v1); and the beautiful, original paper from Gašper Tkačik et al (https://pubmed.ncbi.nlm.nih.gov/20660781/)

Please reach out for questions, comments, and requests!
