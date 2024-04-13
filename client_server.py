import os
import json
import time
import sys

from web3 import Web3
import ipfshttpclient

import model_utils
import pandas as pd
# import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('HTTP://127.0.0.1:7545'))


# Set the default account (replace with the participant's account)
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

df = model_utils.load_data('diabetes.csv')
X_train, X_test, y_train, y_test = model_utils.preprocess_data(df)
X_train, X_test = model_utils.standardize_data(X_train, X_test)

def enroll_participant():
    try:
        tx_hash = contract.functions.enrollParticipant().transact()
        w3.eth.wait_for_transaction_receipt(tx_hash)
        return 'Participant enrolled'
    except Exception as e:
        return f"An error occurred during participant enrollment: {e}"
    
def download_model():
    model_hash = contract.functions.globalModelHash().call()
    client.get(model_hash)
    
    # If the target file already exists, delete it
    if os.path.exists('global_model_downloaded.keras'):
        os.remove('global_model_downloaded.keras')

    os.rename(model_hash, 'global_model_downloaded.keras')
    model = keras.saving.load_model('global_model_downloaded.keras')
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def upload_local_model_update(model):
    model.save('local_model.keras')
    res = client.add('local_model.keras')
    model_hash = res['Hash']
    tx_hash = contract.functions.uploadLocalModelUpdate(model_hash).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Local model update uploaded')

def train_model_locally():
    model = download_model()
    model = model_utils.train_model(model,X_train, y_train, epochs=3)
    upload_local_model_update(model)


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
    enroll_participant()

    wait_while_training_not_started()

    train_model_locally()

    current_block = w3.eth.block_number

    training_task_running = True
    while training_task_running:
      global_model_updated_filter = contract.events.GlobalModelUpdated.create_filter(fromBlock=current_block)
      training_task_completed_filter = contract.events.TrainingTaskCompleted.create_filter(fromBlock=current_block)
      print('Waiting for updates')

      for event in global_model_updated_filter.get_all_entries() + training_task_completed_filter.get_all_entries():
            # print(event)

            if(event['event'] == "GlobalModelUpdated"):
                train_model_locally()
                print('Model updated')
                current_block = w3.eth.block_number+1

            if(event['event'] == "TrainingTaskCompleted"):
                print('Training completed')
                training_task_running = False
                break
      time.sleep(5)  # Adjust the polling interval as needed
      