import time
import json
import os

from web3 import Web3
import ipfshttpclient

import model_utils
import keras
from tensorflow import reduce_mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('HTTP://127.0.0.1:7545'))

# Set the default account (replace with the owner's account)
w3.eth.default_account = w3.eth.accounts[0]

contract_address = None

contract_abi = None

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
client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')

df = model_utils.load_data('diabetes.csv')
X_train, X_test, y_train, y_test = model_utils.preprocess_data(df)
X_train, X_test = model_utils.standardize_data(X_train, X_test)

def aggregate_models(models):
    # Get the weights from the models
    weights = [model.get_weights() for model in models]

    # Calculate the average of the weights
    average_weights = [reduce_mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]

    # Create a new model with the same architecture as the original models
    aggregated_model = keras.models.clone_model(models[0])

    # Set the weights of the new model to the average weights
    aggregated_model.set_weights(average_weights)

    model_utils.evaluate_model(aggregated_model, X_test, y_test)

    return aggregated_model

def initiate_task():
    print('Initiating task')
    model = model_utils.initialize_model()

    # Save the model to a file
    model.save('global_model.keras')

    # Add the file to IPFS
    res = client.add('global_model.keras')
    model_hash = res['Hash']

    # Store the model's CID in the smart contract
    tx_hash = contract.functions.publishTrainingTask(model_hash, "").transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print('Task initiated')

def aggregate_updates():
    # Call the aggregateLocalModelUpdates function in the contract
    local_model_update_hashes = contract.functions.aggregateLocalModelUpdates().call()

    # Download the local model updates from IPFS
    local_model_updates = []
    for hash in local_model_update_hashes:
        client.get(hash)
        if os.path.exists('client_model.keras'):
            os.remove('client_model.keras')
        os.rename(hash, 'client_model.keras')
        model = keras.models.load_model('client_model.keras')
        local_model_updates.append(model)

    # Aggregate the local model updates
    global_model = aggregate_models(local_model_updates)

    # Save the global model to a file
    global_model.save('global_model.keras')

    # Add the file to IPFS
    res = client.add('global_model.keras')
    global_model_hash = res['Hash']

    # Update the global model hash in the contract
    tx_hash = contract.functions.updateGlobalModelHash(global_model_hash).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)

def open_registration():
    tx_hash = contract.functions.openRegistration().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Registration opened for participants')

    second = 0
    while second < 15:
        participant_enrolled_filter = contract.events.ParticipantEnrolled.create_filter(fromBlock="latest")
        print('Waiting for participants to enroll')
        for event in participant_enrolled_filter.get_all_entries():
            # print(event)
            if event['event'] == "ParticipantEnrolled":
                print('Participant enrolled', event['args']['participant'])
        time.sleep(2)
        second += 2


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

    initiate_task()
    open_registration()
    close_registration()
    start_training()

    epoch = 10
    current_block = w3.eth.block_number

    while epoch > 0:
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
        time.sleep(7)  # Adjust the polling interval as needed

    mark_training_complete()
    time.sleep(3)
    print('Training complete')
