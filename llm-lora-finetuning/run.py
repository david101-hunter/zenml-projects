# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from typing import Optional

import click
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML LLM Lora Finetuning project.

Examples:

  \b
  # Run the feature data preparation pipeline
    python run.py --data-pipeline
  
  \b
  # Run the finetuning pipeline
    python run.py --finetuning-pipeline

  \b 
  # Run the merging pipeline
    python run.py --merging-pipeline

  \b
  # Run the evaluation pipeline
    python run.py --eval-pipeline
"""
)
@click.option(
    "--config",
    type=str,
    default=None,
    help="Path to the YAML config file.",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--finetuning-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that finetunes the model.",
)
@click.option(
    "--merging-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that merges the model and adapter.",
)
@click.option(
    "--eval-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that evaluates the model.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    config: Optional[str] = None,
    feature_pipeline: bool = False,
    finetuning_pipeline: bool = False,
    merging_pipeline: bool = False,
    eval_pipeline: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
        no_cache: If `True` cache will be disabled.
    """
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {"enable_cache": not no_cache}
    if config:
        pipeline_args["config_path"] = os.path.join(config_folder, config)

    if feature_pipeline:
        from pipelines.feature_engineering import feature_engineering_pipeline

        feature_engineering_pipeline.with_options(**pipeline_args)()

    if finetuning_pipeline:
        from pipelines.finetuning import finetuning_pipeline

        finetuning_pipeline.with_options(**pipeline_args)()

    if merging_pipeline:
        from pipelines.merge import merge_pipeline

        merge_pipeline.with_options(**pipeline_args)()

    if eval_pipeline:
        from pipelines.eval import eval_pipeline

        eval_pipeline.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
