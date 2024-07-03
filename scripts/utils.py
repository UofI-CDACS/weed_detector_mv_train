import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_model(model, path):
    model.save(path)
