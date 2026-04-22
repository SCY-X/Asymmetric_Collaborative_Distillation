from ._base import Vanilla
from .FitNet import FitNet
from .CC import CC
from .CSD import CSD
from .ROP import ROP
from .RAML import RAML
from .D3 import D3
from .UGD import UGD


distiller_dict = {
    "NONE": Vanilla,
    "FitNet": FitNet,
    "CC": CC,
    "CSD": CSD,
    "ROP": ROP,
    "RAML": RAML,
    "D3": D3,
    "UGD": UGD
}
