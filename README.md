# Safe and Verified AI Project

## An Empirical Study of Supervised Contrastive Loss for Enhancing CNN Robustness

This repository contains the code, experiments, and notebooks used to investigate whether Supervised Contrastive Learning (SCL) can improve the robustness of CNN classifiers against adversarial attacks and formal verification bounds.

Neural networks are widely successful in classification tasks but suffer from limited robustness due to weak feature representations and unstable decision boundaries.  
This project evaluates whether training with Supervised Contrastive Loss (SCL) leads to greater resistance to adversarial perturbations and improved certified robustness via formal verification. More specifically, multiple training strategies were tested on the CIFAR-10 dataset, including Normal training, Adversarial training (PGD) and Certified training (CROWN-IBP), each with and without data augmentation.

It is important to note that all training runs, loss curves, accuracy plots, robustness metrics, and hyperparameter logs are tracked and available in [WandB](https://wandb.ai/lorenzocusin02/Cnn-Verification).
Moreover, final results and plots can be looked at in the [Results Folder](./models_info/results/).

## Requirements

1. Create a `venv` using `python3.10 -m venv venv`
2. Activate the environment using `source ./venv/bin/activate`
3. Install the requirements using `pip install -r requirements.txt`
4. Install alpha-beta-crown using:
    ```sh
    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
    cd alpha-beta-CROWN
    pip install -e .
    cd ..
    ```

## Key Results

- Baseline models deteriorate quickly with increasing adversarial strength.
- Adversarial and certified training improve robustness compared to normal training.
- SCL does not consistently improve robustness across all settings.
- In adversarial regimes, contrastive models sometimes underperform due to excessive input distribution widening.
- Certified contrastive models show minor gains only at very small ε but suffer accuracy loss overall.

Conclusions are detailed in the final report.

## Final Report

The complete analysis, methodology, and experiment results are described in the attached project report:
[“On the Effect of Supervised Contrastive Learning on CNN Robustness”](./report.pdf)
