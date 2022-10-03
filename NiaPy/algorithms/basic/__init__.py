"""Implementation of basic nature-inspired algorithms."""

from NiaPy.algorithms.basic.ba import BatAlgorithm
from NiaPy.algorithms.basic.fa import FireflyAlgorithm
from NiaPy.algorithms.basic.de import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.basic.fpa import FlowerPollinationAlgorithm
from NiaPy.algorithms.basic.gwo import GreyWolfOptimizer
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm
from NiaPy.algorithms.basic.pso import ParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso import BackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_gwo import BackwardGreyWolfOptimizer
from NiaPy.algorithms.basic.B_ba import BackwardBatAlgorithm
from NiaPy.algorithms.basic.B_fpa import BackwardFlowerPollinationAlgorithm
from NiaPy.algorithms.basic.B_pso_with_var import VBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_gwo_with_var import VBackwardGreyWolfOptimizer
from NiaPy.algorithms.basic.B_fpa_with_var import VBackwardFlowerPollinationAlgorithm
from NiaPy.algorithms.basic.B_ba_with_var import VBackwardBatAlgorithm
from NiaPy.algorithms.basic.B_pso_with_state import SVBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_with_var_state_nop import SNoPVBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_op_v import OPVBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.RW_pso import RWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.CW_pso import CWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.LDW_pso import LDWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_CW_pso import BCWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_LDW_pso import BLDWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_RW_pso import BRWParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_attract_bp import TPABackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_with_RS import RSBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_with_RS_TPA import TPARSBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_RS_with_state import RSSBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.B_pso_with_state_var import SRVBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.Levy_pso import LevyFlightParticleSwarmAlgorithm
from NiaPy.algorithms.basic.LF_B_pso import LFBackwardParticleSwarmAlgorithm
from NiaPy.algorithms.basic.Levy_Flight import Levy_Flight
from NiaPy.algorithms.basic.AIWC_pso import AIWCParticleSwarmAlgorithm
from NiaPy.algorithms.basic.IAC_pso import IACParticleSwarmAlgorithm

__all__ = [
    'BatAlgorithm',
    'FireflyAlgorithm',
    'DifferentialEvolutionAlgorithm',
    'FlowerPollinationAlgorithm',
    'GreyWolfOptimizer',
    'GeneticAlgorithm',
    'ArtificialBeeColonyAlgorithm',
    'ParticleSwarmAlgorithm',
    'BackwardParticleSwarmAlgorithm',
    'BackwardGreyWolfOptimizer',
    'BackwardBatAlgorithm',
    'BackwardFlowerPollinationAlgorithm',
    'VBackwardParticleSwarmAlgorithm',
    'VBackwardGreyWolfOptimizer',
    'VBackwardFlowerPollinationAlgorithm',
    'VBackwardBatAlgorithm',
    'SVBackwardParticleSwarmAlgorithm',
    'SNoPVBackwardParticleSwarmAlgorithm',
    'OPVBackwardParticleSwarmAlgorithm',
    'BCWParticleSwarmAlgorithm',
    'BLDWParticleSwarmAlgorithm',
    'BRWParticleSwarmAlgorithm',
    'TPABackwardParticleSwarmAlgorithm',
    'RSBackwardParticleSwarmAlgorithm',
    'TPARSBackwardParticleSwarmAlgorithm',
    'RSSBackwardParticleSwarmAlgorithm',
    'SRVBackwardParticleSwarmAlgorithm',
    'LevyFlightParticleSwarmAlgorithm',
    'LFBackwardParticleSwarmAlgorithm',
    'Levy_Flight',
    'AIWCParticleSwarmAlgorithm',
    'IACParticleSwarmAlgorithm'

]
