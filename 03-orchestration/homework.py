import pandas as pd
from datetime import datetime
from dateutil.relativedelta import *

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import task, flow, get_run_logger
import mlflow, pickle
from mlflow.tracking import MlflowClient


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    mean_duration = df["duration"]
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


@task
def train_model(df, categorical, date):
    with mlflow.start_run():
        logger = get_run_logger()

        train_dicts = df[categorical].to_dict(orient="records")
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        y_train = df.duration

        logger.info(f"The shape of X_train is {X_train.shape}")
        logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)

        mlflow.log_metric("mse", mse)
        logger.info(f"The MSE of training is: {mse}")

        with open(f"models/dv-{date}.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(f"models/dv-{date}.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, f"model-{date}.bin")

        run = mlflow.active_run()
        run_id = run.info.run_id

        return lr, dv, run_id


@task
def run_model(df, categorical, dv, lr, date, run_id):
    client = MlflowClient()

    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")

    model_uri = f"runs:/{run_id}/model"

    mlflow.register_model(model_uri=model_uri, name=f"model-{date}.bin")

    return


@task
def get_paths(date):
    prefered_date = ""
    train_path = "fhv_tripdata_"
    val_path = "fhv_tripdata_"
    if date == None:
        prefered_date_train = datetime.now() - relativedelta(months=2)
        prefered_date_train = prefered_date_train.strftime("%Y-%m")
        prefered_date_val = datetime.now() - relativedelta(months=1)
        prefered_date_val = prefered_date_val.strftime("%Y-%m")
        train_path += prefered_date_train + ".parquet"
        val_path = val_path + prefered_date_val + ".parquet"

        return train_path, val_path
    else:
        prefered_date_train = datetime.strptime(f"{date}", "%Y-%m-%d") - relativedelta(
            months=2
        )
        prefered_date_train = prefered_date_train.strftime("%Y-%m")
        prefered_date_val = datetime.strptime(f"{date}", "%Y-%m-%d") - relativedelta(
            months=1
        )
        prefered_date_val = prefered_date_val.strftime("%Y-%m")
        train_path += prefered_date_train + ".parquet"
        val_path = val_path + prefered_date_val + ".parquet"

        return train_path, val_path


@flow
def main(date="2021-08-15"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("homework-experiment")

    train_path, val_path = get_paths(date).result()

    categorical = ["PUlocationID", "DOlocationID"]

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv, run_id = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr, date, run_id)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="model_training",
    flow=main,
    schedule=CronSchedule(cron="0 9 15 * * "),
    flow_runner=SubprocessFlowRunner(),
)
