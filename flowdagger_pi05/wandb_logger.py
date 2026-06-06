"""Weights and Biases logging helpers: WandBLogger and create_exp_name."""

import os
import datetime
import wandb

import dateutil.tz
from collections import OrderedDict
import numpy as np
from numbers import Number
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """Create a timestamped, semi-unique experiment name."""
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})
    try:
        stats = OrderedDict([
            (name + ' Mean', np.mean(data)),
            (name + ' Std', np.std(data)),
        ])
    except:
        stats = OrderedDict([
            (name + ' Mean', -1),
            (name + ' Std', -1),
        ])
    if not exclude_max_min:
        try:
            stats[name + ' Max'] = np.max(data)
            stats[name + ' Min'] = np.min(data)
        except:
            stats[name + ' Max'] = -1
            stats[name + ' Min'] = -1
    return stats


class WandBLogger(object):
    def __init__(self, wandb_logging, variant, project, experiment_id, output_dir=None, group_name='', team=None):
        self.wandb_logging = wandb_logging
        output_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(output_dir, exist_ok=True)
        if wandb_logging:
            print('wandb using experimentid: ', experiment_id)
            print('wandb using project: ', project)
            print('wandb using group: ', group_name)

            # Credentials come from the standard wandb env / `wandb login`
            # (WANDB_API_KEY, etc.); team/entity is passed in explicitly.
            if "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "run"
            wandb.init(
                config=variant,
                project=project,
                dir=output_dir,
                id=experiment_id,
                settings=wandb.Settings(start_method="thread"),
                group=group_name,
                entity=team
            )
            wandb.define_metric("num_online_trajs")
            wandb.define_metric("env_steps")
            self.output_dir = output_dir

    def log(self, *args, **kwargs):
        if self.wandb_logging:
            wandb.log(*args, **kwargs)

    def log_histogram(self, name, values, step):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        plt.tight_layout()

        plt.hist(np.array(values.flatten()), bins=100)

        canvas.draw()  # draw the canvas, cache the renderer
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        self.log({name: wandb.Image(out_image)}, step=step)
