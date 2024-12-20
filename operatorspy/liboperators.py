import os
import platform
import ctypes
from ctypes import c_int, c_int64, c_uint64, Structure, POINTER
from .data_layout import *
from .devices import *

Device = c_int
Optype = c_int

LIB_OPERATORS_DIR = "INFINI_ROOT"


class TensorDescriptor(Structure):
    _fields_ = [
        ("dt", DataLayout),
        ("ndim", c_uint64),
        ("shape", POINTER(c_uint64)),
        ("pattern", POINTER(c_int64)),
    ]


infiniopTensorDescriptor_t = ctypes.POINTER(TensorDescriptor)


class CTensor:
    def __init__(self, desc, data):
        self.descriptor = desc
        self.data = data


class Handle(Structure):
    _fields_ = [("device", c_int)]


infiniopHandle_t = POINTER(Handle)


# Open operators library
def open_lib():
    def find_library_in_ld_path(library_name):
        ld_library_path = os.environ.get(LIB_OPERATORS_DIR, "")
        paths = ld_library_path.split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, library_name)
            print(full_path)
            if os.path.isfile(full_path):
                return full_path
        return None

    system_name = platform.system()
    # Load the library
    if system_name == "Windows":
        library_path = find_library_in_ld_path("operators.dll")
    elif system_name == "Linux":
        library_path = find_library_in_ld_path("liboperators.so")
    assert (
        library_path is not None
    ), f"Cannot find operators.dll or liboperators.so. Check if {LIB_OPERATORS_DIR} is set correctly."
    lib = ctypes.CDLL(library_path)
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        DataLayout,
    ]
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t), c_int, c_int]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int

    return lib
