# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Controller


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_compatibility_01() -> None:

    ### TEST:
    # 1. tree graph
    # 2. ordern 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure operator and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible

    return None


def test_compatibility_02() -> None:

    ### TEST:
    # 1. tree graph
    # 2. ordern 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure operator and run backward
    controller: Controller = Controller(tensor=GO)
    controller.display_graph()

    return None
