# 全局设置
import argparse, glob, os, logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Path does not exist: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError("No CSV files found.")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split complete.")
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy}")
    return model

def main(args):
    mlflow.sklearn.autolog()
    df = get_csvs_df(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df, 'Diabetic')  # <- 替换为你的目标列
    with mlflow.start_run():
        model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    return parser.parse_args()

if __name__ == "__main__":
    print("\n\n" + "*"*60)
    args = parse_args()
    main(args)
    print("*"*60 + "\n\n")

