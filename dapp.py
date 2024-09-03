import json
import logging

import numpy as np
from cartesi import DApp, Rollup, RollupData, URLRouter
from cartesi.models import _str2hex
from cartesi.wallet.ether import EtherWallet
from ml_logic import DataManager, MachineLearningModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize the DApp
dapp = DApp()
url_router = URLRouter()

# Machine Learning Model and Data Manager
ml_model = MachineLearningModel()
data_manager = DataManager()

# Ether wallet setup
ETHER_PORTAL_ADDRESS = "0xffdbe43d4c855bf7e0f105c400a50857f53ab044"
ether_wallet = EtherWallet(portal_address=ETHER_PORTAL_ADDRESS)
dapp.add_router(ether_wallet)


@dapp.advance()
def handle_advance(rollup: Rollup, data: RollupData) -> bool:
    raw_payload = data.str_payload()
    payload = json.loads(raw_payload)
    command = payload.get("command")

    if command == "ADD_DATA":
        X = np.array(payload["data"]["X"])
        y = np.array(payload["data"]["y"])
        data_manager.add_data(X, y)
        rollup.notice(f"Added {len(X)} new data points")

    elif command == "TRAIN_MODEL":
        X, y = data_manager.get_data()
        if len(X) == 0:
            rollup.notice("No data available for training")
        else:
            message = ml_model.train(X, y)
            metrics = ml_model.evaluate(X, y)
            rollup.notice(f"Training complete. Metrics: {metrics}")

    elif command == "PREDICT":
        X = np.array(payload["data"]["X"])
        predictions = ml_model.predict(X)
        rollup.notice(f"Predictions: {predictions.tolist()}")

    elif command == "SAVE_MODEL":
        model_params = ml_model.save_model()
        rollup.notice(f"Model saved with parameters: {model_params}")

    elif command == "LOAD_MODEL":
        model_params = payload["data"]["model_params"]
        ml_model.load_model(model_params)
        rollup.notice("Model loaded with provided parameters")

    elif command == "CLEAR_DATA":
        data_manager.clear_data()
        rollup.notice("All data cleared")

    else:
        rollup.notice("Invalid command")

    return True


@url_router.inspect("/predict")
def inspect_predict():
    try:
        X, _ = data_manager.get_data()
        if len(X) == 0:
            return {"error": "No data available for prediction"}
        predictions = ml_model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}


@url_router.inspect("/model_params")
def inspect_model_params():
    try:
        model_params = ml_model.save_model()
        return {"model_params": model_params}
    except Exception as e:
        return {"error": str(e)}


@url_router.inspect("/model_metrics")
def inspect_model_metrics():
    try:
        X, y = data_manager.get_data()
        if len(X) == 0:
            return {"error": "No data available for evaluation"}
        metrics = ml_model.evaluate(X, y)
        return {"metrics": metrics}
    except Exception as e:
        return {"error": str(e)}


@url_router.inspect("/data_summary")
def inspect_data_summary():
    try:
        X, y = data_manager.get_data()
        return {
            "num_samples": len(X),
            "num_features": X.shape[1] if len(X) > 0 else 0,
            "X_mean": X.mean(axis=0).tolist() if len(X) > 0 else None,
            "y_mean": y.mean() if len(y) > 0 else None,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    dapp.run()
