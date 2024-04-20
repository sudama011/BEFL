import os
import json
import time
import sys

from web3 import Web3
import ipfshttpclient

import model.model_utils as model_utils
import keras

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('HTTP://127.0.0.1:7545'))


# Set the default account
account_no = 1

if len(sys.argv) > 1:
  account_no = int(sys.argv[1])
w3.eth.default_account = w3.eth.accounts[account_no]

contract_address = None
contract_abi = None

# Load the contract ABI and address
if(os.path.exists('build/contracts/FederatedLearning.json')):
    with open('build/contracts/FederatedLearning.json', 'r') as f:
        contract_abi = json.load(f)
        contract_address = contract_abi["networks"]["5777"]["address"]
        contract_abi = contract_abi["abi"]
else:
    print("Contract ABI file not found")
    exit()


# Create contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Connect to IPFS
client = ipfshttpclient.connect('/dns/127.0.0.1/tcp/5001/http')

directory = f'contributor{account_no}'
X_train, y_train = model_utils.load_data(f"{directory}/local_data.pkl")


def enroll_participant():
    try:
        tx_hash = contract.functions.enrollParticipant().transact()
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print('Participant enrolled successfully')
    except Exception as e:
        if isinstance(e.args[1], dict) and 'reason' in e.args[1]:
            print(e.args[1]['reason'])
        else:
            print(e)
    
def download_model():
    model_hash = contract.functions.globalModelHash().call()
    file_path = f'{directory}/global_model.keras'
    client.get(model_hash, directory)

    if os.path.exists(file_path):
        os.remove(file_path)
    os.rename(f'{directory}/{model_hash}', file_path)

    model = keras.models.load_model(file_path,compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def upload_local_model_update(model):
    file_path = f'{directory}/local_model.keras'
    model.save(file_path)
    res = client.add(file_path)
    model_hash = res['Hash']
    tx_hash = contract.functions.uploadLocalModelUpdate(model_hash).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Local model update uploaded')

def train_model_locally():
    model = download_model()
    model = model_utils.train_model(model,X_train, y_train, epochs=1)
    upload_local_model_update(model)
    print('Model updated')


def wait_while_training_not_started():
    while True:
        event_filter = contract.events.TrainingTaskStarted.create_filter(fromBlock=max(0, w3.eth.block_number - 5))
        print('Waiting for training task to start')
        for event in event_filter.get_all_entries():
            print(event)
            if event['event'] == "TrainingTaskStarted":
                print('Training task started')
                return
        time.sleep(5)  # Adjust the polling interval as needed

if __name__ == '__main__':

    try:
        enroll_participant()

        wait_while_training_not_started()

        train_model_locally()

        current_block = w3.eth.block_number
        training_task_running = True
        
        while training_task_running:

            try:
                global_model_updated_filter = contract.events.GlobalModelUpdated.create_filter(fromBlock=current_block)
                training_task_completed_filter = contract.events.TrainingTaskCompleted.create_filter(fromBlock=current_block)

                for event in global_model_updated_filter.get_all_entries() + training_task_completed_filter.get_all_entries():
                    # print(event)

                    if(event['event'] == "GlobalModelUpdated"):
                        train_model_locally()
                        current_block = w3.eth.block_number+1

                    if(event['event'] == "TrainingTaskCompleted"):
                        print('Training completed')
                        training_task_running = False
                        break
            except Exception as e:
                if isinstance(e.args[1], dict) and 'reason' in e.args[1]:
                    print(e.args[1]['reason'])
                else:
                    print(e)
                exit()
            time.sleep(8)  # Adjust the polling interval as needed
    except Exception as e:
        print(e)
        exit()    