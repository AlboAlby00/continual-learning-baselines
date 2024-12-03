import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss

from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import DER

from models.resnet18 import SingleHeadReducedResNet18
from experiments.utils import set_seed, create_default_args


def der_scifar10(override_args=None):
    """
    """

    args = create_default_args(
        {'cuda': 0, 'alpha': 0.2, 'beta': 0.2,
         'mem_size': 200, 'batch_size_mem': 32, 'lr': 0.03,
         'train_mb_size': 32, 'train_epochs': 20, 'seed': 4}, override_args
    )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # Benchmark
    benchmark = avl.benchmarks.SplitCIFAR10(
        n_experiences=5,
        return_task_id=False
    )

    # Loggers and metrics
    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    # Strategy
    model = SingleHeadReducedResNet18(num_classes=benchmark.n_classes)
    cl_strategy = DER(
        model,
        torch.optim.SGD(model.parameters(), lr=args.lr),
        CrossEntropyLoss(),
        alpha=args.alpha,
        beta=args.beta,
        train_mb_size=args.train_mb_size,
        train_epochs=args.train_epochs,
        batch_size_mem=args.batch_size_mem,
        mem_size=args.mem_size,
        eval_mb_size=32,
        device=device,
        evaluator=evaluation_plugin
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
    return res


if __name__ == '__main__':
    res = der_scifar10()
    print(res)
