#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting
the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    logging.info("Downloading and reading artifact from W&B")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logging.info(f"Downloaded artifact to {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    logging.info("Performing basic cleaning of dataset")
    # Remove samples outside of price range
    len_df_before = len(df)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx]
    len_df_after = len(df)
    logging.info(
        f"Removed {len_df_before - len_df_after} rows from DataFrame for being outside of price range. DataFrame now has {len_df_after} samples.")

    # Remove samples outside of minimum_nights range
    len_df_before = len(df)
    idx = df['minimum_nights'].between(0, args.max_minimum_nights)
    df = df[idx]
    len_df_after = len(df)
    logging.info(
        f"Removed {len_df_before - len_df_after} rows from DataFrame for being outside of minimum_nights range. DataFrame now has {len_df_after} samples.")

    # Change last_review column to be datetime type
    df['last_review'] = pd.to_datetime(df['last_review'])
    logging.info("The last_review feature changed to datetime type")

    # Remove amples outside of NYC boundary
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx]

    # Export cleaned artifact
    # Save processed data to a csv file
    filename = 'clean_sample.csv'
    df.to_csv(filename, index=False)
    logger.info(f'Output cleaned data to csv file named {filename}')

    # Upload to W&B
    logger.info('Upload cleaned data to Weights and Biases as artifact')
    artifact = wandb.Artifact(name=args.output_artifact,
                              type=args.output_type,
                              description=args.output_description
                              )

    artifact.add_file(local_path=filename)
    run.log_artifact(artifact_or_path=artifact)

    # Remove local file
    logger.info('Remove local cleaned file')
    os.remove(filename)

    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A very basic data cleaning workflow")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact stored in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact to be stored in W&B",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the produced artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the produced artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price allowed. Any samples with a lower price will be removed from the dataset.",
        required=True)

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price allowed. Any samples with a higher price will be removed from the dataset.",
        required=True)

    parser.add_argument(
        "--max_minimum_nights",
        type=int,
        help="Maximum value for the minimum_nights feature. Any samples with a higher value will be removed from the dataset.",
        required=True)

    args = parser.parse_args()

    go(args)
