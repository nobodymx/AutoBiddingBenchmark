import atexit
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import os.path as osp
import torch


class Logger:
    def __init__(self, writer, output_fname="progress.txt", log_path="log", model_path="log/bcb/nash_bcb/models/"):
        self.writer = writer
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        self.output_file = open(os.path.join(self.log_path, output_fname), 'w')
        atexit.register(self.output_file.close)
        self.model_path = model_path

    def record(self, tag, scalar_value, global_step, printed=True, end="\n"):
        self.writer.add_scalar(tag, scalar_value, global_step)
        if printed:
            info = f"{tag}: {scalar_value:.3f}, [training_step]: {global_step}"
            print("\033[1;32m [info]\033[0m: " + info, end=end)
            self.output_file.write(info + '\n')

    def info_print(self, info):
        print("\033[1;32m [info]\033[0m: " + info)
        self.output_file.write("[info]: " + info + '\n')

    def warning_print(self, warning):
        print("\033[1;33m [warning]\033[0m: " + warning)
        self.output_file.write("[warning]: " + warning + '\n')

    def error_print(self, error):
        print("\033[1;31m [error]\033[0m: " + error)
        self.output_file.write("[error]: " + error + '\n')

    def stage_print(self, stage):
        print("\033[1;34m [stage]\033[0m: " + stage)
        self.output_file.write("\033[1;34m [stage]\033[0m: " + stage + '\n')

    def record_vector(self, tag, vector_value, global_step, printed=True, outfile=True, end="\n"):
        # self.writer.add_scalars(tag, {str(i): vector_value[i] for i in range(len(vector_value))}, global_step)
        info = tag
        for i in range(len(vector_value)):
            info += str(vector_value[i]) + ", "
        info += "[training_step]:" + str(global_step)
        if printed:
            print("\033[1;32m [info]\033[0m: " + info, end=end)
        if outfile:
            self.output_file.write(info + '\n')


def create_log_and_device(args, config, algo_name="mopo"):
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{config["seed"]}_time_{t0}-{args.task.replace("-", "_")}'
    if args.nebula_mdl:
        log_path = os.path.join(args.task, args.algo_name, log_file)
        log_path = osp.join("/data/oss_bucket_0/log", log_path)
        writer = SummaryWriter(log_dir=os.environ['SUMMARY_DIR'])
    else:
        log_path = os.path.join(config["logdir"], args.task, args.algo_name, log_file)
        writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer, log_path=log_path)
    logger.info_print("log path:" + log_path)

    gpu_enable = True if config["gpu_enable"] else False
    if gpu_enable and torch.cuda.is_available():
        device = torch.device("cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu_id"]
        logger.info_print("setting device: GPU_" + config["gpu_id"])
    else:
        device = torch.device("cpu")
        logger.info_print("setting device: CPU")
    return logger, device
