# Blockchain-based Federated Learning

This project is a decentralized federated learning system built on Ethereum. It uses a smart contract to manage the federated learning process, including participant registration, local model update submission, and model aggregation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [Node.js and npm](https://nodejs.org/en/download/)
- [Truffle](https://www.trufflesuite.com/truffle)
- [Ganache](https://www.trufflesuite.com/ganache)
- [IPFS](https://ipfs.io/)

### Installing

1. Clone the repository:
    ```bash
    git clone repo-url
    cd repo
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. install Truffle globally if you haven't already:
    ```bash
    npm install -g truffle
    ```

4. Start Ganache and ipfs And setup ganache private network
    
5. Compile and migrate the smart contract:
    ```bash
    truffle compile
    truffle migrate
    ```

### Running the tests
1. Run the tests using the following command:
    ```bash
    truffle test
    ```

## Usage

1. Start the central server:
    ```bash
    python central_server.py
    ```

2. Start the participant server in 1 or more terminals:
    ```bash
    python client_server.py 1
    python client_server.py 2
    ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [Strategies for Enhancing Training and Privacy in Blockchain Enabled Federated Learning](https://ieeexplore.ieee.org/abstract/document/9232466)