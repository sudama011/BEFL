const FederatedLearning = artifacts.require("FederatedLearning");

contract("FederatedLearning", (accounts) => {
  let instance;
  const taskPublisher = accounts[0];
  const participant = accounts[1];
  const globalModelHash = "Qm123";
  const modelParameterHash = "Qm456";
  const localModelUpdateHash = "Qm789";

  beforeEach(async () => {
    instance = await FederatedLearning.new({ from: taskPublisher });
  });

  it("should initialize correctly", async () => {
    const actualTaskPublisher = await instance.taskPublisher();
    assert.equal(
      actualTaskPublisher,
      taskPublisher,
      "Task publisher is not set correctly"
    );

    const actualTrainingTaskCompleted = await instance.trainingTaskCompleted();
    assert.equal(
      actualTrainingTaskCompleted,
      false,
      "Training task completed is not initialized correctly"
    );

    const actualRegistrationOpen = await instance.registrationOpen();
    assert.equal(
      actualRegistrationOpen,
      false,
      "Registration open is not initialized correctly"
    );
  });

  it("should allow the task publisher to publish a training task", async () => {
    await instance.publishTrainingTask(globalModelHash, modelParameterHash, {
      from: taskPublisher,
    });

    const actualGlobalModelHash = await instance.globalModelHash();
    assert.equal(
      actualGlobalModelHash,
      globalModelHash,
      "Global model hash is not set correctly"
    );

    const actualModelParameterHash = await instance.modelParameterHash();
    assert.equal(
      actualModelParameterHash,
      modelParameterHash,
      "Model parameter hash is not set correctly"
    );
  });

  it("should allow the task publisher to open and close registration", async () => {
    await instance.openRegistration({ from: taskPublisher });
    let actualRegistrationOpen = await instance.registrationOpen();
    assert.equal(
      actualRegistrationOpen,
      true,
      "Registration open is not set correctly"
    );

    await instance.closeRegistration({ from: taskPublisher });
    actualRegistrationOpen = await instance.registrationOpen();
    assert.equal(
      actualRegistrationOpen,
      false,
      "Registration open is not set correctly"
    );
  });

  it("should enroll a participant", async () => {
    await instance.openRegistration({ from: taskPublisher });
    await instance.enrollParticipant({ from: participant });

    const participantData = await instance.participants(participant);

    assert.equal(
      participantData.registered,
      true,
      "Participant is not registered correctly"
    );
  });

  it("should upload a local model update", async () => {
    await instance.openRegistration({ from: taskPublisher });
    await instance.enrollParticipant({ from: participant });
    await instance.closeRegistration({ from: taskPublisher });
    await instance.startTrainingTask({ from: taskPublisher });
    await instance.uploadLocalModelUpdate(localModelUpdateHash, {
      from: participant,
    });

    const participantData = await instance.participants(participant);

    assert.equal(
      participantData.localModelUpdateHash,
      localModelUpdateHash,
      "Local model update hash is not set correctly"
    );
  });

  it("should not allow a participant to enroll when registration is closed", async () => {
    try {
      await instance.enrollParticipant({ from: participant });
      assert.fail("Expected revert not received");
    } catch (error) {
      const revertFound = error.message.search("revert") >= 0;
      assert(revertFound, `Expected "revert", got ${error} instead`);
    }
  });

  it("should not allow a participant to upload a local model update when training task has not started", async () => {
    try {
      await instance.uploadLocalModelUpdate(localModelUpdateHash, {
        from: participant,
      });
      assert.fail("Expected revert not received");
    } catch (error) {
      const revertFound = error.message.search("revert") >= 0;
      assert(revertFound, `Expected "revert", got ${error} instead`);
    }
  });
});
