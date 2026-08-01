from hydra.core.config_store import ConfigStore

from .algo.base import BaseAlgoConfig
from .algo.ctrlsr import *
from .algo.diffsr import *
from .algo.dppo import DPPOConfig
from .algo.dppo_scratch import DPPOFromScratchConfig
from .algo.dpmd import DPMDConfig
from .algo.fpo import FPOConfig
from .algo.fpopp import FPOPPConfig
from .algo.genpo import GenPOConfig
from .algo.idem import IDEMConfig
from .algo.policyflow import PolicyFlowConfig
from .algo.ppo import PPOConfig
from .algo.qsm import QSMConfig
from .algo.sac import SACConfig
from .algo.sdac import SDACConfig
from .algo.td3 import TD3Config
from .algo.td7 import TD7Config
from .dmc_config import Config as DMCConfig
from .hb_config import Config as HBConfig
from .mujoco_config import Config as MUJOCOConfig
from .onpolicy_dmc_config import Config as OnPolicyDMCConfig
from .onpolicy_isaaclab_config import Config as OnPolicyIsaacLabConfig
from .onpolicy_mujoco_config import Config as OnPolicyMUJOCOConfig

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="dmc_config" + _DEF_SUFFIX, node=DMCConfig)
cs.store(name="hb_config" + _DEF_SUFFIX, node=HBConfig)
cs.store(name="mujoco_config" + _DEF_SUFFIX, node=MUJOCOConfig)
cs.store(name="onpolicy_dmc_config" + _DEF_SUFFIX, node=OnPolicyDMCConfig)
cs.store(name="onpolicy_isaaclab_config" + _DEF_SUFFIX, node=OnPolicyIsaacLabConfig)
cs.store(name="onpolicy_mujoco_config" + _DEF_SUFFIX, node=OnPolicyMUJOCOConfig)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    "sac": SACConfig,
    "sdac": SDACConfig,
    "td3": TD3Config,
    "td7": TD7Config,
    "dpmd": DPMDConfig,
    "qsm": QSMConfig,
    "idem": IDEMConfig,
    "dppo": DPPOConfig,
    "dppo_scratch": DPPOFromScratchConfig,
    "ppo": PPOConfig,
    "policyflow": PolicyFlowConfig,
    "fpo": FPOConfig,
    "fpopp": FPOPPConfig,
    "genpo": GenPOConfig,
    "ctrlsr_td3": CtrlSRTD3Config,
    "diffsr_td3": DiffSRTD3Config,
    "diffsr_ld": DiffSRLDConfig,
    "diffsr_qsm": DiffSRQSMConfig,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name + _DEF_SUFFIX, node=cfg)
