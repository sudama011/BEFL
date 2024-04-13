// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    address public taskPublisher;
    string public globalModelHash;
    string public modelParameterHash;
    uint public updatesSubmitted;
    bool public trainingTaskStarted;
    bool public trainingTaskCompleted;
    bool public registrationOpen;

    struct Participant {
        bool registered;
        string localModelUpdateHash;
    }

    address[] public participantAddresses;
    mapping(address => Participant) public participants;

    modifier onlyTaskPublisher() {
        require(
            msg.sender == taskPublisher,
            "Only the task publisher can call this function"
        );
        _;
    }

    constructor() {
        taskPublisher = msg.sender;
        trainingTaskCompleted = false;
        registrationOpen = false;
    }

    function publishTrainingTask(
        string memory _globalModelHash,
        string memory _modelParameterHash
    ) public onlyTaskPublisher {
        globalModelHash = _globalModelHash;
        modelParameterHash = _modelParameterHash;
        updatesSubmitted = 0;
        trainingTaskCompleted = false;
        registrationOpen = false;
    }

    function openRegistration() public onlyTaskPublisher {
        registrationOpen = true;
    }

    function closeRegistration() public onlyTaskPublisher {
        registrationOpen = false;
    }

    event TrainingTaskStarted();
    function startTrainingTask() public onlyTaskPublisher {
        require(
            registrationOpen == false,
            "Cannot start training task while registration is open"
        );
        require(
            trainingTaskCompleted == false,
            "Cannot start training task after it has been completed"
        );
        trainingTaskStarted = true;
        emit TrainingTaskStarted();
    }

    event ParticipantEnrolled(address indexed participant);
    event ParticipantDeregistered(address indexed participant);
    function enrollParticipant() public {
        require(registrationOpen, "Registration is currently closed");

        require(
            !participants[msg.sender].registered,
            "This address is already registered as a participant"
        );
        participants[msg.sender] = Participant(true, "");
        participantAddresses.push(msg.sender);
        emit ParticipantEnrolled(msg.sender);
    }

    function deregisterParticipant(
        address _participantAddress
    ) public onlyTaskPublisher {
        delete participants[_participantAddress];
        emit ParticipantDeregistered(_participantAddress);
    }

    event AllUpdatesSubmitted();
    function uploadLocalModelUpdate(
        string memory _localModelUpdateHash
    ) public {
        require(trainingTaskStarted, "Training task has not started yet");
        require(
            participants[msg.sender].registered,
            "Only registered participants can submit updates"
        );
        require(
            !trainingTaskCompleted,
            "Training task has already been completed"
        );
        participants[msg.sender].localModelUpdateHash = _localModelUpdateHash;
        updatesSubmitted++;
        if (updatesSubmitted >= participantAddresses.length) {
            emit AllUpdatesSubmitted();
        }
    }

    function aggregateLocalModelUpdates()
        public
        onlyTaskPublisher
        returns (string[] memory)
    {
        require(
            updatesSubmitted >= participantAddresses.length,
            "All participants must submit their updates before aggregation"
        );
        string[] memory localModelUpdateHashes = new string[](
            participantAddresses.length
        );
        for (uint i = 0; i < participantAddresses.length; i++) {
            localModelUpdateHashes[i] = participants[participantAddresses[i]]
                .localModelUpdateHash;
            participants[participantAddresses[i]].localModelUpdateHash = ""; // Reset update hash
        }
        updatesSubmitted = 0; // Reset the counter
        return localModelUpdateHashes;
    }

    event GlobalModelUpdated(
        address indexed taskPublisher,
        string newGlobalModelHash
    );
    function updateGlobalModelHash(
        string memory _globalModelHash
    ) public onlyTaskPublisher {
        globalModelHash = _globalModelHash;
        emit GlobalModelUpdated(taskPublisher, _globalModelHash);
    }

    event TrainingTaskCompleted();
    function markTrainingTaskAsCompleted() public onlyTaskPublisher {
        trainingTaskCompleted = true;
        emit TrainingTaskCompleted();
    }
}
