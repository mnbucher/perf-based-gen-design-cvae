# Performance-Based Generative Design for Parametric Modeling of Engineering Structures Using Deep Conditional Generative Models

[ [ Paper ] ](https://www.sciencedirect.com/science/article/pii/S0926580523003886) – Published in Automation in Construction Journal, December 2023

## Abstract

Parametric Modeling, Generative Design, and Performance-Based Design have gained increasing attention in the AEC field as a way to create a wide range of design variants while focusing on performance attributes rather than building codes. However, the relationships between design parameters and performance attributes are often very complex, resulting in a highly iterative and unguided process. In this paper, we argue that a more goal-oriented design process is enabled by an inverse formulation that starts with performance attributes instead of design parameters. A Deep Conditional Generative Design workflow is proposed that takes a set of performance attributes and partially defined design features as input and produces a complete set of design parameters as output. A model architecture based on a Conditional Variational Autoencoder is presented along with different approximate posteriors, and evaluated on four different case studies. Compared to Genetic Algorithms, our method proves superior when utilizing a pre-trained model.

## Code

The model architectures are located at ```src/models/cvae_XXX.py``` where the XXX postfix indicates the particular architecture choice with different posteriors and architecture tweaks. Please check the paper for a corresponding ablation study. We found ```cvae_05_nf.py``` to be the strongest, while ```cvae_08_nf_yxpart.py``` also shows good performance while offering more controllability over the generation process (although sacrificing some performance). The latter will be an interesting direction to pursue for further future work. If you have any questions about the code, don't hesitate to reach out to the main author (Martin; me) via email. You can find my email on my personal website [www.mnbucher.com](www.mnbucher.com)
