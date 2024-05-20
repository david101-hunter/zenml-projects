from zenml import pipeline, step, get_step_context
from zenml.client import Client
from mlflow.tracking import MlflowClient, artifact_utils
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from mlflow.deployments import MLFlowDeploymentConfig
@step
def deploy_model() -> Optional[MLFlowDeploymentService]:
    # Deploy a model using the MLflow Model Deployer
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    experiment_tracker = zenml_client.active_stack.experiment_tracker
    # Let's get the run id of the current pipeline
    mlflow_run_id = experiment_tracker.get_run_id(
        experiment_name=get_step_context().pipeline_name,
        run_name=get_step_context().run_name,
    )
    # Once we have the run id, we can get the model URI using mlflow client
    experiment_tracker.configure_mlflow()
    client = MlflowClient()
    model_name = "model" # set the model name that was logged
    model_uri = artifact_utils.get_artifact_uri(
        run_id=mlflow_run_id, artifact_path=model_name
    )
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name: str = "mlflow-model-deployment-example",
        description: str = "An example of deploying a model using the MLflow Model Deployer",
        pipeline_name: str = get_step_context().pipeline_name,
        pipeline_step_name: str = get_step_context().step_name,
        model_uri: str = model_uri,
        model_name: str = model_name,
        workers: int = 1,
        mlserver: bool = False,
        timeout: int = 300,
    )
    service = model_deployer.deploy_model(mlflow_deployment_config)
    return service