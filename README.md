## Environment configuration (Mac)

1. Create a `venv` using `python3.10 -m venv venv`;
2. Activate the environment using `source ./venv/bin/activate`
3. Install the requirements using `pip install -r requirements.txt`
4. Install alpha-beta-crown using:
    ```sh
    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
    cd alpha-beta-CROWN
    pip install -e .
    cd ..
    ```

---

# Results

The Adversarial Contrastive model is wrong, do not consider it.

- Accuracy Train -> Test (without data augmentation and adversarial examples). THe accuracy on Train and Validation is on [WandB](https://wandb.ai/lorenzocusin02/Cnn-Verification/workspace?nw=nwuserlorenzocusin02)
	- With Augmentation
		- Normal Model: 80.56% -> 77.04%
		- Contrastive Model: 82.95% -> 79.32%
		- Adversarial Model: 71.72% -> 69.14%
		- Adversarial Contrastive: 79.19% -> 76.93%
	- No Augmentation
		- Normal Model: 98.84% -> 67.51%
		- Contrastive Model: 99.04% -> 67.13%
		- Adversarial Model: 88.80% -> 68.76%
		- Adversarial Contrastive: 98.29% -> 68.57%

- PGD Attack Success Rate: (PLOT in the models_info folder)
    - Eps: 1/255
      - With Augmentation
        - Normal Model ASR: 15.47%;
        - Contrastive Model ASR: 14.41%;
        - Adversarial Model ASR: 4.40%;
        - Adversarial Contrastive ASR: 21.73%;
      - No Augmentation
        - Normal Model ASR: 34.13%;
        - Contrastive Model ASR: 42.62%;
        - Adversarial Model ASR: 5.13%;
        - Adversarial Contrastive ASR: 24.17%;
    - Eps: 2/255
      - With Augmentation
        - Normal Model ASR: 34.21%;
    	- Contrastive Model ASR: 31.31%;
    	- Adversarial Model ASR: 9.60%;
    	- Adversarial Contrastive ASR: 44.71%;
      - No Augmentation
        - Normal Model ASR: 64.30%;
    	- Contrastive Model ASR: 72.09%;
    	- Adversarial Model ASR: 10.92%;
    	- Adversarial Contrastive ASR: 47.64%;
    - Eps: 4/255
      - With Augmentation
        - Normal Model ASR: 66.69%;
    	- Contrastive Model ASR: 62.74%;
    	- Adversarial Model ASR: 18.79%;
    	- Adversarial Contrastive ASR: 80.97%;
      - No Augmentation
        - Normal Model ASR: 92.11%;
    	- Contrastive Model ASR: 95.96%;
    	- Adversarial Model ASR: 22.81%;
    	- Adversarial Contrastive ASR: 78.61%;
    - Eps: 8/255
      - With Augmentation
        - Normal Model ASR: 94.28%;
    	- Contrastive Model ASR: 91.11%;
    	- Adversarial Model ASR: 37.70%;
    	- Adversarial Contrastive ASR: 99.18%;
      - No Augmentation
        - Normal Model ASR: 99.00%;
    	- Contrastive Model ASR: 99.81%;
    	- Adversarial Model ASR: 45.61%;
    	- Adversarial Contrastive ASR: 96.52%;
    - Eps: 16/255
      - With Augmentation
        - Normal Model ASR: 99.52%;
    	- Contrastive Model ASR: 98.08%;
    	- Adversarial Model ASR: 59.50%;
    	- Adversarial Contrastive ASR: 99.99%;
      - No Augmentation
        - Normal Model ASR: 99.73%;
    	- Contrastive Model ASR: 99.97%;
    	- Adversarial Model ASR: 65.90%;
    	- Adversarial Contrastive ASR: 99.24%;

- ABCrown executed on 50 samples:
  - EPSILON 1/255 con 10 sample {'Normal Model': {'safe-incomplete': 8, 'unsafe-pgd': 1, 'safe': 1}, 'Contrastive Model': {'unsafe-pgd': 1, 'safe-incomplete': 5, 'safe': 4}, 'Adversarial Model': {'safe-incomplete': 9, 'unsafe-pgd': 1}, 'Adversarial Contrastive': {'safe': 7, 'safe-incomplete': 2, 'unsafe-pgd': 1}}
  - 