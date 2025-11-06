import json
import mlflow
import logging
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/Aryanupadhyay23/mlops-mini-project.mlflow')
dagshub.init(repo_owner='Aryanupadhyay23', repo_name='mlops-mini-project', mlflow=True)

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logger.debug('Model info loaded from %s', file_path)
    return model_info


def log_model_as_artifact(model_info: dict, model_name: str):
    """Log the model as an artifact (instead of using the registry)."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        mlflow.log_artifact("models/model.pkl", artifact_path=f"{model_name}_logged_model")
        logger.debug(f'Model {model_name} logged successfully as artifact from {model_uri}')
    except Exception as e:
        logger.error('Error during model logging: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "my_model"
        log_model_as_artifact(model_info, model_name)
    except Exception as e:
        logger.error('Failed to complete the model logging process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

