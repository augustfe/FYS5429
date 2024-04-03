# %%
import jax.numpy as np
import matplotlib.pyplot as plt

from xpinn import XPINN
from utils import data_path, model_path, fig_path
import plotutils
from NavierStokes_utils.model_predict import NSpredict


# %%
activation = np.tanh


NoDecomp_file_test = data_path / "NavierStokes" / "test_NoDecomp.json"
TwoBoxDecomp_file_test = data_path / "NavierStokes" / "test_TwoBoxDecomp.json"

xpinnTwoBox = XPINN(TwoBoxDecomp_file_test, activation)
single_pinn = XPINN(NoDecomp_file_test, activation)


# %% [markdown]
# ## Model paths

# %% [markdown]
# No decomp models

# %%

plt.axis("equal"
         )

for i, pinn in enumerate(single_pinn.PINNs):
    checkout = pinn.interior
    plt.scatter(checkout[:, 0], checkout[:, 1], label=f"PINN {i}")

plt.legend()


# %%
no_decomp_path = model_path / "NavierStokes" / "NoDecomp" / "laminar"

# models\NavierStokes\NoDecomp\laminar\ADAM_20000_iter_9layer\model0
ND_adam_20000_iter_9layer = no_decomp_path / "ADAM_20000_iter_9layer" / "model0"

# models/NavierStokes/NoDecomp/laminar/Pretrained_ADAM_20000_iter_8layer
ND_adam_pretrained_20000_iter_8layer = no_decomp_path / \
    "Pretrained_ADAM_20000_iter_8layer" / "model1"

# models\NavierStokes\NoDecomp\laminar\NoCylinder\ADAM_20000_iter_8layer
ND_adam_no_cylinder_20000_iter_8layer = no_decomp_path / \
    "NoCylinder" / "ADAM_20000_iter_8layer"

# models\NavierStokes\single_pinn\laminar\Adam_20000_iter_7layer_model0
ND_adam_20000_iter_good_model = model_path / "NavierStokes" / \
    "single_pinn" / "laminar" / "Adam_20000_iter_7layer_model0"

# models\NavierStokes\single_pinn\laminar\Best_single_pinn
ND_best = model_path / "NavierStokes" / \
    "single_pinn" / "laminar" / "Best_single_pinn"


# %% [markdown]
# ### Decomp models


# %% [markdown]
# #### Two box decomp

# %%

plt.axis("equal"
         )

for i, pinn in enumerate(xpinnTwoBox.PINNs):
    checkout = pinn.interior
    plt.scatter(checkout[:, 0], checkout[:, 1], label=f"PINN {i}")

plt.legend()

# %% [markdown]
# Two box model paths

# %%
two_box_model_path = model_path / "NavierStokes" / "Decomp_2"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=4000
TB_4000_iter = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=4000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000
TB_10000_iter = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_20.0_1.0_1.0_iterations=10000
TB_10000_iter_20_inflow_weight = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_20.0_1.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_20.0_20.0_1.0_iterations=10000
TB_10000_iter_20_inflow_wall_weight = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_20.0_20.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_20.0_40.0_1.0_iterations=10000
TB_10000_iter_20_inflow_wall_weight_40 = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_20.0_40.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_20.0_80.0_1.0_iterations=10000
TB_10000_iter_20_inflow_wall_weight_80 = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_20.0_80.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_40.0_20.0_1.0_iterations=10000
TB_10000_iter_40_inflow_wall_weight_20 = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_40.0_20.0_1.0_iterations=10000"


# models\NavierStokes\Decomp_2\laminar_decomp_train_500_5000_v2_1.0_1.0_1.0_iterations=10000
TB_10000_iter_500_5000 = two_box_model_path / \
    "laminar_decomp_train_500_5000_v2_1.0_1.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_500_5000_v2_20.0_40.0_1.0_iterations=10000
TB_10000_iter_500_5000_20_40 = two_box_model_path / \
    "laminar_decomp_train_500_5000_v2_20.0_40.0_1.0_iterations=10000"


# models\NavierStokes\Decomp_2\laminar_decomp_train_500_5000_v2_20.0_20.0_1.0_iterations=10000
TB_10000_iter_500_5000_20_20 = two_box_model_path / \
    "laminar_decomp_train_500_5000_v2_20.0_20.0_1.0_iterations=10000"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000_Neumann
TB_10000_iter_Neumann = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000_Neumann"

# models\NavierStokes\Decomp_2\laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000_right_emphasis
TB_10000_iter_right_emphasis = two_box_model_path / \
    "laminar_decomp_train_400_2100_v2_1.0_1.0_1.0_iterations=10000_right_emphasis"


# %% [markdown]
# #### Load prefered models

# %%

# Change these and run cells if you want to load a different model
# single_pinn_model = ND_adam_20000_iter_9layer
single_pinn_model = ND_best


# TwoBox_model = TB_10000_iter_500_5000_20_20
# TwoBox_model = TB_10000_iter_500_5000_20_40
# TwoBox_model = TB_10000_iter_500_5000
# TwoBox_model = TB_10000_iter_20_inflow_wall_weight_80
# TwoBox_model = TB_10000_iter_20_inflow_wall_weight_40
# TwoBox_model = TB_10000_iter_Neumann
TwoBox_model = TB_10000_iter_right_emphasis

# single_pinn_model_str = "ND_adam_20000_iter_9layer"
single_pinn_model_str = "ND_best"


# TwoBox_model_str = "TB_10000_iter_500_5000_20_20"
# TwoBox_model_str = "TB_10000_iter_500_5000_20_40"
# TwoBox_model_str = "TB_10000_iter_500_5000"
# TwoBox_model_str = "TB_10000_iter_20_inflow_wall_weight_40"
# TwoBox_model_str = "TB_10000_iter_Neumann"
TwoBox_model_str = "TB_10000_iter_right_emphasis"


# %%

single_pinn.load_model(single_pinn_model)


xpinnTwoBox.load_model(TwoBox_model)

# %% [markdown]
# #### Losses

# %%
save_path_NoDecomp = fig_path / "NavierStokes" / \
    "NoDecomp" / f"{single_pinn_model_str}" / "losses"

save_path_TBDecomp = fig_path / "NavierStokes" / \
    "TwoBoxDecomp" / f"{TwoBox_model_str}" / "losses"


# XPINNtitle = "XPINN solution"
# PINNtitle = "PINN solution"

PINNlossTitle = "PINN losses over 10 000 iterations"

TwoBoxlossTitle = "XPINN losses over 10 000 iterations"


save_path_NoDecomp.mkdir(parents=True, exist_ok=True)

save_path_TBDecomp.mkdir(parents=True, exist_ok=True)

# %%
plotutils.plot_losses(
    a_losses=single_pinn.losses,
    n_iter=10000,
    title=PINNlossTitle,
    savepath=save_path_NoDecomp,
    save_name=f"PINN_losses",
)

# %%
plotutils.plot_losses(
    a_losses=xpinnTwoBox.losses,
    n_iter=10000,
    title=TwoBoxlossTitle,
    savepath=save_path_TBDecomp,
    save_name=f"TwoBox_decomp_losses",
)

# %% [markdown]
# ### Navier Stokes results

# %% [markdown]
# Savepaths and names for single pinn navier stokes results

# %%
No_decomp_solution_save_path = fig_path / "NavierStokes" / \
    "NoDecomp" / f"{single_pinn_model_str}" / "solution"


# create_path
No_decomp_solution_save_path.mkdir(parents=True, exist_ok=True)

# %%
points, flow, flow_magitude, pressure, streamfunction = NSpredict(
    single_pinn, NoDecomp_file_test)

# %%
tot_points = np.concatenate(points)

# %%
"""
def plot_navier_stokes(
    points: Array,
    val: Array,
    title: str,
    savepath: Path,
    save_name: str,
    clim: tuple = None,
):
"""

# %%
# clim_bench_flow = [0, 0.405]
# clim_bench_pressure = [0.0115, 0.131]
# clim_stram = [0.0396, 0.0424]

# %%
tot_flow = np.concatenate(flow_magitude)
clim_flow = (tot_flow.min(), tot_flow.max())

plotutils.plot_navier_stokes(
    tot_points,
    tot_flow,
    "PINN Flow Magnitude",
    No_decomp_solution_save_path,
    "flow_magnitude_no_decomp",
    clim=clim_flow,
)

# %%
tot_pressure = np.concatenate(pressure)
clim_pressure = (tot_pressure.min(), tot_pressure.max())

plotutils.plot_navier_stokes(
    tot_points,
    tot_pressure,
    "PINN Pressure",
    No_decomp_solution_save_path,
    "pressure_no_decomp",
    clim=clim_pressure,
)

# %%
tot_streamfunction = np.concatenate(streamfunction)
clim_stream = (tot_streamfunction.min(), tot_streamfunction.max())

plotutils.plot_navier_stokes(
    tot_points,
    tot_streamfunction,
    "PINN Streamfunction",
    No_decomp_solution_save_path,
    "streamfunc_no_decomp",
    clim=clim_stream,
)

# %%
points, flow, flow_magitude, pressure, streamfunction = NSpredict(
    xpinnTwoBox, TwoBoxDecomp_file_test)

# %%
tot_points_x = np.concatenate(points)

# %% [markdown]
# Two box decomp save paths

# %%
two_box_solution_save_path = fig_path / "NavierStokes" / \
    "TwoBoxDecomp" / f"{TwoBox_model_str}" / "solution"


# make paths
two_box_solution_save_path.mkdir(parents=True, exist_ok=True)

# %%
tot_flow_x = np.concatenate(flow_magitude)
clim_flow_x = (tot_flow_x.min(), tot_flow_x.max())

plotutils.plot_navier_stokes(
    tot_points_x,
    tot_flow_x,
    "XPINN Flow Magnitude",
    two_box_solution_save_path,
    "flow_magnitude_two_box",
    clim=clim_flow,
)

# %%
tot_pressure_x = np.concatenate(pressure)
clim_pressure = (tot_pressure_x.min(), tot_pressure_x.max())

plotutils.plot_navier_stokes(
    tot_points_x,
    tot_pressure_x,
    "XPINN Pressure",
    two_box_solution_save_path,
    "pressure_two_box",
    clim=clim_pressure,
)

# %%
tot_streamfunction_x = np.concatenate(streamfunction)
clim_stream = (tot_streamfunction_x.min(), tot_streamfunction_x.max())

plotutils.plot_navier_stokes(
    tot_points_x,
    tot_streamfunction_x,
    "XPINN Streamfunction",
    two_box_solution_save_path,
    "streamfunc_two_box",
    clim=clim_stream,

)
