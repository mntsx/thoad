# Standard Library Dependencies
import gc
from typing import Type

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction

# Extended Autograd Functions
from thoad.autodifferentiation.internals.accumulation import AccumulateGradX
from thoad.autodifferentiation.internals.activation import CeluXBackward0
from thoad.autodifferentiation.internals.activation import GeluXBackward0
from thoad.autodifferentiation.internals.activation import LeakyReluXBackward0
from thoad.autodifferentiation.internals.activation import PreluKernelXBackward0
from thoad.autodifferentiation.internals.activation import ReluXBackward0
from thoad.autodifferentiation.internals.activation import RreluWithNoiseXBackward0
from thoad.autodifferentiation.internals.activation import SiluXBackward0
from thoad.autodifferentiation.internals.activation import LogSoftmaxXBackward0
from thoad.autodifferentiation.internals.activation import SoftmaxXBackward0
from thoad.autodifferentiation.internals.activation import SoftplusXBackward0
from thoad.autodifferentiation.internals.condition import WhereXBackward0
from thoad.autodifferentiation.internals.contraction import AddmmXBackward0
from thoad.autodifferentiation.internals.contraction import BmmXBackward0
from thoad.autodifferentiation.internals.contraction import DotXBackward0
from thoad.autodifferentiation.internals.contraction import MmXBackward0
from thoad.autodifferentiation.internals.contraction import MvXBackward0
from thoad.autodifferentiation.internals.exponentiation import PowXBackward0
from thoad.autodifferentiation.internals.indexation import CloneXBackward0
from thoad.autodifferentiation.internals.mathematic import AbsXBackward0
from thoad.autodifferentiation.internals.mathematic import LogXBackward0
from thoad.autodifferentiation.internals.mathematic import NegXBackward0
from thoad.autodifferentiation.internals.mathematic import SigmoidXBackward0
from thoad.autodifferentiation.internals.multiplication import DivXBackward0
from thoad.autodifferentiation.internals.multiplication import MulXBackward0
from thoad.autodifferentiation.internals.multiplication import ProdXBackward0
from thoad.autodifferentiation.internals.refiguration import PermuteXBackward0
from thoad.autodifferentiation.internals.refiguration import SqueezeXBackward0
from thoad.autodifferentiation.internals.refiguration import SqueezeXBackward1
from thoad.autodifferentiation.internals.refiguration import SqueezeXBackward2
from thoad.autodifferentiation.internals.refiguration import TXBackward0
from thoad.autodifferentiation.internals.refiguration import TransposeXBackward0
from thoad.autodifferentiation.internals.refiguration import UnsqueezeXBackward0
from thoad.autodifferentiation.internals.refiguration import ViewXBackward0
from thoad.autodifferentiation.internals.summation import AddXBackward0
from thoad.autodifferentiation.internals.summation import SubXBackward0
from thoad.autodifferentiation.internals.summation import SumXBackward0
from thoad.autodifferentiation.internals.summation import SumXBackward1
from thoad.autodifferentiation.internals.trigonometry import CosXBackward0
from thoad.autodifferentiation.internals.trigonometry import CoshXBackward0
from thoad.autodifferentiation.internals.trigonometry import SinXBackward0
from thoad.autodifferentiation.internals.trigonometry import SinhXBackward0
from thoad.autodifferentiation.internals.trigonometry import TanXBackward0
from thoad.autodifferentiation.internals.trigonometry import TanhXBackward0


def acquire_gfn_map() -> dict[Type, Type]:

    ### Typings & definitions
    aux: Tensor
    xfn_type: Type[ExtendedAutogradFunction]
    mapper: dict[Type, Type] = dict()

    ### Instantiate auxiliary tensors
    TA: Tensor = torch.zeros(size=(1,), requires_grad=True)
    TB: Tensor = torch.zeros(size=(1, 1), requires_grad=True)
    TC: Tensor = torch.zeros(size=(1, 1, 1), requires_grad=True)
    # TD: Tensor = torch.zeros(size=(2,), requires_grad=True)
    # IDX: Tensor = torch.zeros(size=(1,), dtype=torch.long)

    ### ACCUMULATION
    gfn_type = type(torch.sum(TA).grad_fn.next_functions[0][0])
    xfn_type = AccumulateGradX
    mapper[gfn_type] = xfn_type

    ### CONDITION
    # torch.where, Tensor.where
    aux = torch.where(condition=(TB > 0), input=TB, other=TB)
    xfn_type = WhereXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### EXPONENTIATION
    # torch.pow, Tensor.pow (scalar exponent)
    aux = torch.pow(input=TB, exponent=2)
    xfn_type = PowXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.pow, Tensor.pow (tensor exponent)
    # aux = torch.pow(input=TB, exponent=TB)
    # xfn_type = PowXBackward1
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.sqrt, Tensor.sqrt
    # aux = torch.sqrt(input=TB)
    # xfn_type = SqrtXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.exp, Tensor.exp
    # aux = torch.exp(input=TB)
    # xfn_type = ExpXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type

    ### INDEXATION
    # torch.clone, Tensor.clone
    aux = torch.clone(input=TA)
    xfn_type = CloneXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # [int], torch.select, Tensor.select
    # aux = torch.select(input=TA, dim=0, index=0)
    # xfn_type = SelectXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # [:]
    # aux = TA[:]
    # xfn_type = SliceXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # [Tensor]
    # aux = TA[TA > 0]
    # xfn_type = IndexXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.index_select, Tensor.index_select
    # aux = torch.index_select(input=TA, dim=0, index=IDX)
    # xfn_type = IndexSelectXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.index_put, Tensor.index_put_
    # aux = torch.index_put(input=TA, indices=(IDX,), values=TA)
    # xfn_type = IndexPutXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.masked_select, Tensor.masked_select
    # aux = torch.masked_select(input=TA, mask=(TA >= 0.0))
    # xfn_type = MaskedSelectXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.masked_scatter, Tensor.masked_scatter
    # aux = torch.masked_scatter(input=TA, mask=(TA >= 0.0), source=TA)
    # xfn_type = MaskedScatterXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.gather, Tensor.gather
    # aux = torch.gather(input=TA, dim=0, index=IDX)
    # xfn_type = GatherXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.scatter, Tensor.scatter_
    # aux = torch.scatter(input=TA, dim=0, index=TA.long(), src=TA)
    # xfn_type = ScatterXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.take, Tensor.take
    # aux = torch.take(input=TA, index=TA.long())
    # xfn_type = TakeXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.put, Tensor.put
    # aux = torch.put(input=TA, index=TA.long(), source=TA)
    # xfn_type = PutXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type

    ### LOSS
    # # torch.nn.MSELoss, torch.nn.functional.mse_loss
    # aux = torch.nn.functional.mse_loss(input=TA, target=TA)
    # xfn_type = MseLossXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.nn.SmoothL1Loss, torch.nn.functional.smooth_l1_loss
    # aux = torch.nn.functional.smooth_l1_loss(input=TA, target=TA)
    # xfn_type = SmoothL1LossXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.nn.BCELoss, torch.nn.functional.binary_cross_entropy
    # aux = torch.nn.functional.binary_cross_entropy(input=TA, target=TA)
    # xfn_type = BinaryCrossEntropyXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.nn.BCEWithLogitsLoss, torch.nn.functional.binary_cross_entropy_with_logits
    # aux = torch.nn.functional.binary_cross_entropy_with_logits(input=TA, target=TA)
    # xfn_type = BinaryCrossEntropyWithLogitsXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type

    ### LINEAR UNITS
    # torch.nn.CeLU, torch.nn.functional.celu
    aux = torch.nn.functional.celu(input=TA)
    xfn_type = CeluXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.nn.ELU, torch.nn.functional.elu, torch.nn.SELU, torch.nn.functional.selu
    # aux = torch.nn.functional.elu(input=TA)
    # xfn_type = EluXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.GeLU, torch.nn.functional.gelu
    aux = torch.nn.functional.gelu(input=TA)
    xfn_type = GeluXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.nn.GLU, torch.nn.functional.glu
    # aux = torch.nn.functional.glu(input=TD)
    # xfn_type = GluXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.LeakyReLU, torch.nn.functional.leaky_relu
    aux = torch.nn.functional.leaky_relu(input=TA)
    xfn_type = LeakyReluXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.PReLU, torch.nn.functional.prelu
    aux = torch.nn.functional.prelu(input=TA, weight=TA)
    xfn_type = PreluKernelXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
    xfn_type = ReluXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.RReLU, torch.nn.functional.rrelu
    aux = torch.nn.functional.rrelu(input=TA)
    xfn_type = RreluWithNoiseXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.SiLU, torch.nn.functional.silu
    aux = torch.nn.functional.silu(input=TA)
    xfn_type = SiluXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### MATRIX MULTIPLICATION
    # torch.addmm, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.addmm(input=TB, mat1=TB, mat2=TB)
    xfn_type = AddmmXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.bmm, Tensor.bmm
    aux = torch.bmm(input=TC, mat2=TC)
    xfn_type = BmmXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # @, torch.dot, Tensor.dot
    aux = torch.dot(input=TA, tensor=TA)
    xfn_type = DotXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = MmXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # @, torch.mv
    aux = torch.mv(input=TB, vec=TA)
    xfn_type = MvXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### PRODUCTS
    # /, torch.div, Tensor.div, Tensor.div_
    aux = torch.div(input=TB, other=TB)
    xfn_type = DivXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # *, torch.mul, torch.multiply, Tensor.mul, Tensor.mul_
    aux = torch.mul(input=TB, other=TB)
    xfn_type = MulXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.prod, Tensor.prod (all dims)
    aux = torch.prod(input=TB)
    xfn_type = ProdXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.prod, Tensor.prod (along dim=1)
    # aux = torch.prod(input=TB, dim=1)
    # xfn_type = ProdXBackward1
    # mapper[type(aux.grad_fn)] = xfn_type

    ### RESHAPE
    # # torch.expand, Tensor.expand, Tensor.expand_as
    # aux = TA.expand(size=(1,))
    # xfn_type = ExpandXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.permute, Tensor.permute, Tensor.T
    aux = torch.permute(input=TB, dims=(0, 1))
    xfn_type = PermuteXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.view, Tensor.view, Tensor.view_as ~ torch.reshape, Tensor.reshape
    aux = TA.view(size=(1, 1))
    xfn_type = ViewXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.repeat, Tensor.repeat
    # aux = TA.repeat(repeats=(1,))
    # xfn_type = RepeatXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (no dim)
    aux = TC.squeeze()
    xfn_type = SqueezeXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dim=0)
    aux = TC.squeeze(dim=0)
    xfn_type = SqueezeXBackward1
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dims=(0,1))
    aux = TC.squeeze(dim=(0, 1))
    xfn_type = SqueezeXBackward2
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.t, Tensor.t
    aux = torch.t(input=TB)
    xfn_type = TXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.transpose, Tensor.transpose
    aux = torch.transpose(input=TB, dim0=0, dim1=1)
    xfn_type = TransposeXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.unsqueeze, Tensor.unsqueeze
    aux = TA.unsqueeze(dim=0)
    xfn_type = UnsqueezeXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### SOFTENING
    # torch.softmax, torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(input=TA, dim=0)
    xfn_type = SoftmaxXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.LogSoftmax, torch.nn.functional.log_softmax
    aux = torch.nn.functional.log_softmax(input=TA, dim=0)
    xfn_type = LogSoftmaxXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.Softplus, torch.nn.functional.softplus
    aux = torch.nn.functional.softplus(input=TA)
    xfn_type = SoftplusXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### SUMMATIONS
    # +, torch.add, Tensor.add
    aux = torch.add(input=TB, other=TB)
    xfn_type = AddXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sub, Tensor.subtract
    aux = torch.sub(input=TB, other=TB)
    xfn_type = SubXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sum, Tensor.sum (all dims)
    aux = torch.sum(input=TB)
    xfn_type = SumXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sum, Tensor.sum (dim=(1,))
    aux = torch.sum(input=TB, dim=(1,))
    xfn_type = SumXBackward1
    mapper[type(aux.grad_fn)] = xfn_type

    ### TRIGONOMETRY
    # torch.sin
    aux = torch.sin(input=TB)
    xfn_type = SinXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.cos
    aux = torch.cos(input=TB)
    xfn_type = CosXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.tan
    aux = torch.tan(input=TB)
    xfn_type = TanXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sinh
    aux = torch.sinh(input=TB)
    xfn_type = SinhXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.cosh
    aux = torch.cosh(input=TB)
    xfn_type = CoshXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.tanh
    aux = torch.tanh(input=TB)
    xfn_type = TanhXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    ### MORE MATH
    # torch.abs
    aux = torch.abs(input=TB)
    xfn_type = AbsXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.neg
    aux = torch.neg(input=TB)
    xfn_type = NegXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.mean (all dims)
    # aux = torch.mean(input=TB)
    # xfn_type = MeanXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # # torch.mean (dim=1)
    # aux = torch.mean(input=TB, dim=1)
    # xfn_type = MeanXBackward1
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.log
    aux = torch.log(input=TB)
    xfn_type = LogXBackward0
    mapper[type(aux.grad_fn)] = xfn_type
    # # torch.special.xlogy
    # aux = torch.special.xlogy(input=TB, other=TB)
    # xfn_type = XlogyXBackward0
    # mapper[type(aux.grad_fn)] = xfn_type
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = SigmoidXBackward0
    mapper[type(aux.grad_fn)] = xfn_type

    gc.collect()

    return mapper
