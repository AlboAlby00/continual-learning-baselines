import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss

from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import SER

from models import MLP
from experiments.utils import set_seed, create_default_args


def lamaml_scifar100(override_args=None):
    """
    # 43: 87.75
    """

    args = create_default_args(
        {'cuda': 0, 'alpha': 0.1, 'beta': 0.1,
         'mem_size': 100, 'batch_size_mem': 10, 'lr': 0.01,
         'train_mb_size': 10, 'train_epochs': 1, 'seed': 43}, override_args
    )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # Benchmark
    benchmark = avl.benchmarks.SplitMNIST(
        n_experiences=5, 
        return_task_id=False)

    # Loggers and metrics
    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    # Strategy
    model = MLP()
    cl_strategy = SER(
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
    res = lamaml_scifar100()
    print(res)
