from DataLogger.datalogger import DataLogger
from Config.generate_log_data_config import config


def run():
    datalogger = DataLogger(config["basic_config"]["num_agents"])
    datalogger.load()


