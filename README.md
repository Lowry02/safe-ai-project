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

- Accuracy
	- Augmentation
		- Normal Model: 84.08% -> 79.69%
		- Contrastive Model: 82.96% -> 79.27%
		- Adversarial Model: 75.14% -> 72.49%
		- Adversarial Contrastive: 51.76% -> 51.29%
	- No Augmentation
		- Normal Model: 98.84% -> 67.69%
		- Contrastive Model: 99.04% -> 67.60%
		- Adversarial Model: 88.80% -> 68.53%
		- Adversarial Contrastive: 79.79% -> 67.76%