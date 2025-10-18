from azure.ai.ml import Input, MLClient, Output, command, dsl, load_component
from azure.identity import InteractiveBrowserCredential

# ---------------------------------------------------------------------------
# Azure ML workspace configuration
ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="CV6-2025",
)

# ---------------------------------------------------------------------------
# Shared settings
environment_name = "iris"
environment_version = "0.6.0"
compute_target_name = "cloud"
component_path = "./src/utils/"

print(f"Using environment {environment_name} version {environment_version}")
env = ml_client.environments.get(environment_name, environment_version)

# ---------------------------------------------------------------------------
# Register and save components
train_cmd = ml_client.components.create_or_update(
    command(
        name="train",
        display_name="Train model",
        description="Train model with data from a predefined data asset.",
        inputs={"data": Input(type="uri_folder", description="Data asset URI")},
        outputs={"model": Output(type="uri_folder", mode="rw_mount")},
        code=component_path,
        command=(
            "python model_training.py "
            "--patch_size 256 "
            "--learning_rate 0.0001 "
            "--decay_steps 100 "
            "--decay_rate 0.9 "
            "--staircase True "
            "--epochs 30 "
            "--batch_size 16 "
            "--data ${{inputs.data}} "
            "--model_path ${{outputs.model}}"
        ),
        environment=env,
    )
)

evaluate_cmd = ml_client.components.create_or_update(
    command(
        name="evaluate",
        display_name="Evaluate model",
        description="Evaluate model with data from a predefined data asset.",
        inputs={
            "data": Input(type="uri_folder", description="Data asset URI"),
            "model": Input(type="uri_folder", description="Trained model path"),
        },
        outputs={"metrics": Output(type="uri_folder", mode="rw_mount")},
        code=component_path,
        command=(
            "python model_evaluation.py "
            "--patch_size 256 "
            "--data ${{inputs.data}} "
            "--model_path ${{inputs.model}}"
        ),
        environment=env,
    )
)

register_cmd = ml_client.components.create_or_update(
    command(
        name="register",
        display_name="Register model",
        description="Register the trained model.",
        inputs={"model": Input(type="uri_folder", description="Model URI")},
        outputs={"registered_model": Output(type="uri_folder", mode="rw_mount")},
        code=component_path,
        command=(
            "python register.py "
            "--model_path ${{inputs.model}} "
            "--registered_model_path ${{outputs.registered_model}}"
        ),
        environment=env,
    )
)

# ---------------------------------------------------------------------------
# Load components for use in pipeline
t_train = load_component(source=train_cmd.id)
t_evaluate = load_component(source=evaluate_cmd.id)
t_register = load_component(source=register_cmd.id)

# ---------------------------------------------------------------------------
# Pipeline definition
@dsl.pipeline(
    name="iris_pipeline",
    description="Pipeline for training, evaluating, and registering a PlantSeg model.",
)
def iris_pipeline_func(data_input):
    train_job = t_train(data=data_input).set_compute(compute_target_name)
    evaluate_job = t_evaluate(
        data=data_input,
        model=train_job.outputs.model
    ).set_compute(compute_target_name)
    register_job = t_register(
        model=train_job.outputs.model
    ).set_compute(compute_target_name)

    return {
        "trained_model": train_job.outputs.model,
        "evaluation_results": evaluate_job.outputs.metrics,
        "registered_model": register_job.outputs.registered_model
    }

# ---------------------------------------------------------------------------
# Submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    iris_pipeline_func(
        data_input=Input(type="uri_folder", path="azureml:my_dataset_name:1")
    ),
    experiment_name="iris_pipeline"
)
print(f"Pipeline job submitted: {pipeline_job.name}")

# ---------------------------------------------------------------------------
# Wait for completion
pipeline_job.wait()
print(f"Pipeline job completed with status: {pipeline_job.status}")
# ---------------------------------------------------------------------------