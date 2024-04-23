import time
import json
import os

from web3 import Web3
import ipfshttpclient

import utils.model_utils as model_utils
import keras

w3 = Web3(Web3.HTTPProvider('HTTP://127.0.0.1:7545'))

# Set the default account (replace with the owner's account)
w3.eth.default_account = w3.eth.accounts[0]

contract_address = None

contract_abi = None

if(os.path.exists('contracts/FederatedLearning.json')):
    with open('contracts/FederatedLearning.json', 'r') as f:
        contract_abi = json.load(f)
        contract_address = contract_abi["networks"]["5777"]["address"]
        contract_abi = contract_abi["abi"]
else:
    print("Contract ABI file not found")
    exit()

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')

directory = 'server_data'
X_test, y_test = model_utils.load_data(f"{directory}/local_data.pkl")


def initiate_task():
    print('Initiating task')
    model = model_utils.initialize_simple_model()
    model.save(f'{directory}/global_model.keras')
   
    # Add the file to IPFS
    res = client.add(f'{directory}/global_model.keras')
    model_hash = res['Hash']

    # Store the model's CID in the smart contract
    tx_hash = contract.functions.publishTrainingTask(model_hash, "" ).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print('Task initiated')

def aggregate_updates():
    local_model_update_hashes = contract.functions.aggregateLocalModelUpdates().call()

    # Download the local model updates from IPFS
    local_model_updates = []
    for hash in local_model_update_hashes:
        client.get(hash, directory)
        if os.path.exists(f'{directory}/client_model.keras'):
            os.remove(f'{directory}/client_model.keras')
        os.rename(f'{directory}/{hash}', f'{directory}/client_model.keras')
        model = keras.models.load_model(f'{directory}/client_model.keras')
        local_model_updates.append(model)

    global_model = model_utils.aggregate_models(local_model_updates)
    model_utils.evaluate_model(global_model, X_test, y_test)
    global_model.save(f'{directory}/global_model.keras')

    res = client.add(f'{directory}/global_model.keras')
    global_model_hash = res['Hash']

    tx_hash = contract.functions.updateGlobalModelHash(global_model_hash).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)

def open_registration():
    tx_hash = contract.functions.openRegistration().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Registration opened for participants')

    second = 25
    while second > 0:
        participant_enrolled_filter = contract.events.ParticipantEnrolled.create_filter(fromBlock="latest")
        for event in participant_enrolled_filter.get_all_entries():
            if event['event'] == "ParticipantEnrolled":
                print('Participant enrolled', event['args']['participant'])
        time.sleep(3)
        second -= 3


def close_registration():
    tx_hash = contract.functions.closeRegistration().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Registration closed')

def start_training():
    tx_hash = contract.functions.startTrainingTask().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Training started now participants can submit their updates')

def mark_training_complete():
    tx_hash = contract.functions.markTrainingTaskAsCompleted().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Training task marked as complete')

if __name__ == '__main__':
    try:
        initiate_task()
        open_registration()
        close_registration()
        start_training()

        epoch = 10
        current_block = w3.eth.block_number

        while epoch > 0:
            try:
                # print(epoch, 'Waiting for updates')

                event_filter = contract.events.AllUpdatesSubmitted.create_filter(fromBlock=current_block)
                for event in event_filter.get_all_entries():
                    # print(event)

                    if(event['event']== "AllUpdatesSubmitted"):
                        print(epoch, 'All updates submitted')
                        aggregate_updates()
                        print(epoch, 'Updates aggregated')
                        epoch -= 1
                    if epoch == 0:
                        break
                    current_block = w3.eth.block_number+1
            except Exception as e:
                print(f"Error in training loop: {e}")
                break

            time.sleep(10)  # Adjust the polling interval as needed

        mark_training_complete()
        print('Training complete')
    except Exception as e:
        print(e)
        print('Error occurred')
        exit()