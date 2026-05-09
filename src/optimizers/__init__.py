from .baselines import train_sgd, train_adam, train_projected_sgd
from .frank_wolfe import train_sfw_paper, train_fw_vc
from .constraints import project_onto_l1_ball, l1_lmo, frank_wolfe_gap
