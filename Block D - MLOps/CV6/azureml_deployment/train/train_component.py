"""Submit a retraining job using a registered model."""

from azure.ai.ml import Input,Output, MLClient, load_component
from azure.identity import InteractiveBrowserCredential


def main() -> None:
    """Load training component and submit job to Azure ML."""
    ml_client = MLClient(
        InteractiveBrowserCredential(),
        subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
        resource_group_name="buas-y2",
        workspace_name="CV6-2025",
    )

    train_component_func = load_component(source="./azureml_deployment/train/train_component.yml")
    train_component = ml_client.components.create_or_update(train_component_func)

    job = train_component(
        model=Input(type="uri_file", path="azureml_deployment:iris-model:1"),
        data=Input(type="uri_folder", path="azureml_deployment:preprocessed-patch-dataset:1"),
        learning_rate=0.0001,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
        epochs=30,
        patch_size=256,
        batch_size=16,
        outputs=dict(model=Output(type="uri_folder", mode="rw_mount")),
    )

    submitted_job = ml_client.jobs.create_or_update(
        job, experiment_name="iris_train"
    )
    print(f"Training job submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
