import yaml

def load_configuration():
    configuration = None

    with open('./configurations/config.yaml', 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)

    return configuration

configuration = load_configuration()
