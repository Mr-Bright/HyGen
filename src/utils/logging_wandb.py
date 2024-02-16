import logging
import socket
from collections import defaultdict

import numpy as np
import wandb

import wandb


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False
        # 保证wandb step单增
        self.max_t = -1
        self.base_t = 0

        self.stats = defaultdict(lambda: [])

    def setup_wandb(self, directory_name, config):
        # Import here so it doesn't have to be installed if you don't use it
        run_name = ""
        if getattr(config, "use_traj_encoder", False):
            run_name = "forward"
        elif getattr(config, "use_bidirection_traj_encoder", False):
            run_name = "bidirection"
        else:
            run_name = "state"

        run_name += "_"
        if getattr(config, "train_hybrid", False):
            run_name += "hybrid-"
            run_name += getattr(config, "hybrid_mode")
        else:
            run_name += "offline"

        run_name += "_"
        if config.cql_loss_mode == "no":
            run_name += "no_cql"
        elif config.cql_loss_mode == "fix":
            run_name += "fix_cql"
        elif config.cql_loss_mode == "dynamic":
            run_name += "dynamic_cql"

        run_name += "_"
        run_name += config.run_name
        run_name += "_" + socket.gethostname()
        config.run_name = run_name

        wandb.init(
            project="hybrid_MARL", dir=directory_name, name=run_name, config=config
        )
        self.use_wandb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_wandb:
            if t >= self.max_t:
                self.max_t = t
            else:
                self.base_t = self.max_t
            t += self.base_t

            wandb.log({key: value}, step=t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def log_histogram(self, key, value, t):
        if t >= self.max_t:
            self.max_t = t
        else:
            self.base_t = self.max_t

        t += self.base_t
        wandb.log({key: wandb.Histogram(value)}, step=t)

    # def log_embedding(self, key, value):

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10}\t Episode: {:>10}\n".format(
            self.stats["episode"][-1][0], self.stats["episode"][-1][1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"

        self.console_logger.info(log_str)

        def finish(self):
            if self.use_wandb:
                wandb.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
