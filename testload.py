import ctypes
import pathlib

libname = pathlib.Path().absolute() / "libadd.so"
c_lib = ctypes.CDLL(libname)
c_lib.add.restype = ctypes.c_float

print(c_lib.add(2, ctypes.c_float(3.4)))
