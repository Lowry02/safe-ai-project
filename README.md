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