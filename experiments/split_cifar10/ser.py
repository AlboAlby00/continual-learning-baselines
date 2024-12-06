import avalanche as avl
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised.buffer_ser import SER
from avalanche.training.plugins import LRSchedulerPlugin

from avalanche.models.resnet18 import ResNet18
from experiments.utils import set_seed, create_default_args


def ser_scifar10(override_args=None):
    # reduced resnet18: 0.4928
    # resnet 0.5183
    # IDK 0.6
    # New buffer 0.6663

    args = create_default_args(
        {
            "cuda": 0,
            "alpha": 0.2,
            "beta": 0.2,
            "mem_size": 200,
            "batch_size_mem": 32,
            "lr": 0.03,
            "train_mb_size": 32,
            "train_epochs": 20,
            "seed": 24,
        },
        override_args,
    )

    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # Benchmark
    fixed_class_order = np.arange(10)
    benchmark = avl.benchmarks.SplitCIFAR10(
        n_experiences=5,
        return_task_id=False,
        class_ids_from_zero_in_each_exp=False,
        shuffle=True,
        fixed_class_order=fixed_class_order
    )

    # Loggers and metrics
    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    # Strategy
    model = ResNet18(nclasses=benchmark.n_classes)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [15], gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler)
    cl_strategy = SER(
        model,
        optim,
        CrossEntropyLoss(),
        alpha=args.alpha,
        beta=args.beta,
        train_mb_size=args.train_mb_size,
        train_epochs=args.train_epochs,
        batch_size_mem=args.batch_size_mem,
        mem_size=args.mem_size,
        eval_mb_size=8,
        device=device,
        evaluator=evaluation_plugin,
        plugins=[scheduler_plugin],
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
    return res


if __name__ == "__main__":
    res = ser_scifar10()
    print(res)
