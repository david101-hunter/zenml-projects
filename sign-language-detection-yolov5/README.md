# Detect and recognize the American Sign Language alphabet on real-time image using Yolov5 and ZenML

**Problem statement**: One of the most anticipated capabilities of Machine Learning and AI is to help people with disabilities. The deaf community cannot do what most of the population take for granted and are often placed in degrading situations due to these challenges they face every day, in this Zenfile (project) will see how computer vision can be utilized to create a model that can bridge the gap for the deaf and hard of hearing by learning American Sign Language and be able to understand the meaning of each sign.
To so This project will use ZenML to create a pipeline that will train a model to detect and recognize the American Sign Language alphabet on real-time image using Yolov5 MLFlow and Vertex AI Platform.

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers the build, track and deploy a computer vision pipeline using some of the most popular tools in the industry.

- By offering you a framework and template to base your own work on.
- By using a custom code Object Detection algorithm called [Yolov5](https://github.com/ultralytics/yolov5)
- By integrating with tools like [MLflow](https://mlflow.org/) to track the hyperparameters and metrics of the model.
- By allowing you to train your model on [Google Vertex AI Platform](https://cloud.google.com/vertex-ai) with minimal effort.

Note : This project is based on [Interactive ABC's with American Sign Language](https://github.com/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5)
the main difference is that this project is using ZenML to create a pipeline that will train a model to detect and recognize the American Sign Language alphabet on real-time image using Yolov5 MLFlow and Vertex AI Platform.

## :notebook: Explanation of the project

In order to build a model that can detect and recognize the American Sign Language alphabet on real-time image we will need to do the following steps:

1. Download the dataset from [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-alphabet)
2. Augment the training and valdiation sets using [Albumentations](https://albumentations.ai/)
3. Train the model using a pretrained model from [Yolov5](https://github.com/ultralytics/yolov5) while tracking the hyperparameters and metrics using [MLflow](https://docs.zenml.io/component-gallery/experiment-trackers/mlflow) within a GPU environment by laverging [Google Vertex AI Step Operator](https://docs.zenml.io/component-gallery/step-operators/gcloud-vertexai) stack component.
4. Load model in a different pipeline that deploys the model using [BentoML]() and provided ZenML integration.
5. Create an inference pipeline that will use the deployed model to detect and recognize the American Sign Language alphabet on test images from the first pipeline.

## Requirements

In order to follow this tutorial, you need to have the following software
installed on your local machine:

* [Python](https://www.python.org/) (version 3.7-3.9)
* [Docker](https://www.docker.com/) installed on your machine
* [GCloud CLI](https://cloud.google.com/sdk/docs/install) installed on your machine and authenticated
* [Remote ZenML Server](https://docs.zenml.io/getting-started/deploying-zenml#deploying-zenml-in-the-cloud-remote-deployment-of-the-http-server-and-database) A Remote Deployment of the ZenML HTTP server and Database

### :rocket: Remote ZenML Server

For Advanced use cases where we have a remote orchestrator or step operators such as Vertex AI
or to share stacks and pipeline information with team we need to have a separated non local remote ZenML Server that it can be accessible from your
machine as well as all stack components that may need access to the server.
[Read more information about the use case here](https://docs.zenml.io/getting-started/deploying-zenml)

In order to achieve this there are two different ways to get access to a remote ZenML Server.

1. Deploy and manage the server manually on [your own cloud](https://docs.zenml.io/getting-started/deploying-zenml)/
2. Sign up for [ZenML Cloud](https://zenml.io/cloud-signup) and get access to a hosted
   version of the ZenML Server with no setup required.

### :snake: Setup Python Environment

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenfiles.git
git submodule update --init --recursive
cd zenfiles/sign-language-detection-yolov5
pip install -r requirements.txt
pip install -r yolov5/requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you 
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to  [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/getting-started/deploying-zenml), but first you must install the optional dependencies for the ZenML server:

```bash
zenml connect --url=$ZENML_SERVER_URL
zenml init
```

### 👣  Step-by-Step on how to set up your GCP project

I will show how to create Google Cloud resources for this project using `gcloud cli`. [Follow this](https://cloud.google.com/sdk/docs/install) if you don't have it set up.

#### 1. Make sure you are in the correct GCP project

List the current configurations and check that `project_id` is set to your GCP project:

```shell
gcloud config list
```

If not, use:

```shell
gcloud config set project <PROJECT_ID>
```

#### 2. Set permissions to create and manage Vertex AI custom jobs and to access data from BigQuery

Create a service account:

```shell
gcloud iam service-accounts create <NAME>

# Example:
gcloud iam service-accounts create zenml-sa
```

Grant permission to the service account:

```shell
gcloud projects add-iam-policy-binding <PROJECT_ID> --member="serviceAccount:<SA-NAME>@<PROJECT_ID>.iam.gserviceaccount.com" --role=<ROLE>

# Example:
gcloud projects add-iam-policy-binding zenml-vertex-ai --member="serviceAccount:zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com" --role=roles/storage.admin
gcloud projects add-iam-policy-binding zenml-vertex-ai --member="serviceAccount:zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com" --role=roles/aiplatform.admin
```

Generate a key file:

```shell
gcloud iam service-accounts keys create <FILE-NAME>.json --iam-account=<SA-NAME>@<PROJECT_ID>.iam.gserviceaccount.com

# Example:
gcloud iam service-accounts keys create credentials.json --iam-account=zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com
```

#### 3. Create a GCP bucket

Vertex AI and ZenML will use this bucket for output of any artifacts from the training run:

```shell
gsutil mb -l <REGION> gs://bucket-name

# Example:
gsutil mb -l europe-west1 gs://zenml-bucket
```

#### 4. Configure and enable Container Registry in GCP

ZenML will use this registry to push your job images that Vertex will use.

a) [Enable](https://cloud.google.com/container-registry/docs) Container Registry


b) [Authenticate](https://cloud.google.com/container-registry/docs/advanced-authentication) your local `docker` cli with your GCP container registry:

```shell
docker pull busybox
docker tag busybox gcr.io/<PROJECT-ID/busybox
docker push gcr.io/<PROJECT-ID>/busybox
```

#### 5. [Enable](https://console.cloud.google.com/marketplace/product/google/aiplatform.googleapis.com?q=search&referrer=search&project=cloudguru-test-project) `Vertex AI API`

To be able to use custom Vertex AI jobs, you first need to enable their API inside Google Cloud console.


### 👣  Set up the components required for ZenML stack

Set a GCP bucket as your artifact store:

```shell
zenml artifact-store register <NAME> --type=gcp --path=<GCS_BUCKET_PATH>

# Example:
zenml artifact-store register gcp-store --type=gcp --path=gs://zenml-bucket
```

Create a Vertex step operator:

```shell
zenml step-operator register <NAME> \
    --type=vertex \
    --project=<PROJECT-ID> \
    --region=<REGION> \
    --machine_type=<MACHINE-TYPE> \
    --accelerator_type=<ACCELERATOR-TYPE> \
    --accelerator_count=<ACCELERATOR-COUNT> \
    --service_account_path=<SERVICE-ACCOUNT-KEY-FILE-PATH>

# Example:
zenml step-operator register vertex \
    --type=vertex \
    --project=zenml-core \
    --region=europe-west1 \
    --machine_type=n1-standard-4 \
    --accelerator_type=NVIDIA_TESLA_K80 \
    --accelerator_count=1 \
    --service_account_path=credentials.json
```

List of [available machines](https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types)

Register a container registry:

```shell
zenml container-registry register <NAME> --type=default --uri=gcr.io/<PROJECT-ID>/<IMAGE>

# Example:
zenml container-registry register gcr_registry --type=default --uri=gcr.io/zenml-vertex-ai/busybox
```

Register a Remote MLFlow tracking server:

```shell
zenml tracking-server register <NAME> --type=mlflow --uri=<MLFLOW-TRACKING-SERVER-URI> --tracking_username=<USERNAME> --tracking_password=<PASSWORD>

# Example:
zenml tracking-server register mlflow --type=mlflow --uri=http://mlflow_zenml_yolo:5000 --tracking_username=admin --tracking_password=admin
```

Register the new stack (change names accordingly):

```shell
zenml stack register vertex_mlflow_stack \
    -o default \
    -c gcr_registry \
    -a gcp-store \
    -s vertex \
    -e mlflow
```

View all your stacks: `zenml stack list`

Activate the stack:

```shell
zenml stack set vertex_training_stack
```
## 📙 Resources & References

We had written a blog that explains this project in-depth: 

If you'd like to watch the video that explains the project, you can watch the [video](https://youtu.be/L3_pFTlF9EQ).


### Training Pipeline

The training pipeline is made up of the following steps:

- `data_loader.py`: Loads the data from the Roboflow platforms using the given API key and saves it to the artifact store as dictionary that contains all information about each set by using the `zenml.artifacts.DatasetArtifact` class.
- `train_augmenter.py`: Loads the training set data from the artifact store and performs data augmentation using the `albumentations` library. It then saves the augmented data to the artifact store as a `zenml.artifacts.DatasetArtifact` class.
- `vald_augmenter.py`: Loads the validation set data from the artifact store and performs data augmentation using the `albumentations` library. It then saves the augmented data to the artifact store as a `zenml.artifacts.DatasetArtifact` class.
- `trainer.py`: Loads the augmented training and validation data from the artifact store and trains the model using the `yolov5` library in a custom Vertex AI job. which track the training process using the `mlflow` library. It then saves the trained model to the artifact store as a `zenml.artifacts.ModelArtifact` class.
 
### Deployment Pipeline

The Deployment pipeline is made up of the following steps:

- `model_loader.py`: Loads the trained model from the previously trained pipeline and saves it locally.
- `deployment_triggeer.py`: Triggers the deployment process once the model is loaded locally.
- `bento_builder.py`: Builds a BentoML bundle from the model and saves it to the artifact store and passes it to the next step, which is the `bento_deployer.py`.
- `bento_deployer.py`: Deploys the BentoML bundle to the Vertex AI endpoint.

### Inference Pipeline

The Inference pipeline is made up of the following steps:

- `inference_loader.py`: Loads the Test set data from the first step of the training pipeline and save it locally.
- `prediction_service_loader.py`: Loads the ZenML prediction service in order to make predictions on the test set data.
- `predictor.py`: Runs the prediction service on the test set data and print the results.

# 📜 References

Documentation on [Step Operators](https://docs.zenml.io/component-gallery/step-operators)

Example of [Step Operators](https://github.com/zenml-io/zenml/tree/main/examples/step_operator_remote_training)

More on [Step Operators](https://blog.zenml.io/step-operators-training/)

Documentation on how to create a GCP [service account](https://cloud.google.com/docs/authentication/getting-started#create-service-account-gcloud)

ZenML CLI [documentation](https://apidocs.zenml.io/latest/cli/)