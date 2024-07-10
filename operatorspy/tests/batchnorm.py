from ctypes import c_float, c_void_p
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
)

from operatorspy.tests.test_utils import get_args
import torch


def batchNorm(a,eps,mean,var):
    input_dtype = a.dtype
    return (
        torch.nn.functional.batch_norm(a.to(torch.float32),running_mean=mean,running_var=var)
    )


def test(lib, descriptor, torch_device):
    a = torch.rand((1, 3, 5,5), dtype=torch.float16).to(torch_device)
    b = torch.rand((1, 3, 5,5), dtype=torch.float16).to(torch_device)
    mean = torch.zeros((3), dtype=torch.float16).to(torch_device)
    var = torch.ones((3), dtype=torch.float16).to(torch_device)
    beta = 0.0
    alpha = 1.0
    eps = 1e-6

    ans = batchNorm(a,eps,mean,var)
    lib.batchNorm(
        descriptor,
        to_tensor(b, lib),
        to_tensor(a, lib),
        c_float(eps),
        None,None,None
    )
    assert torch.allclose(b, ans, atol=0, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createBatchNormDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyBatchNormDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    descriptor = lib.createBatchNormDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroyBatchNormDescriptor(descriptor)

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createBatchNormDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroyBatchNormDescriptor(descriptor)

if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createBatchNormDescriptor.restype = c_void_p
    lib.destroyBatchNormDescriptor.argtypes = [c_void_p]
    lib.batchNorm.argtypes = [
        c_void_p,
        CTensor,
        CTensor,
        c_float,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
