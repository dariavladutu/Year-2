"""Submit evaluation job to Azure ML."""

from azure.ai.ml import MLClient, load_component
from azure.ai.ml.entities import Input
from azure.identity import InteractiveBrowserCredential


def main() -> None:
    """Load evaluate component and submit job to Azure ML."""
    ml_client = MLClient(
        InteractiveBrowserCredential(),
        subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
        resource_group_name="buas-y2",
        workspace_name="CV6-2025",
    )

    evaluate_component = load_component(source="evaluate_component.yml")

    job = evaluate_component(
        model=Input(
            type="uri_file",
            path="azureml:iris-model:1",
        ),
        data=Input(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/val_data/",
        ),
        patch_size=256,
    )

    ml_client.jobs.create_or_update(job, experiment_name="evaluate-model")


if __name__ == "__main__":
    main()
