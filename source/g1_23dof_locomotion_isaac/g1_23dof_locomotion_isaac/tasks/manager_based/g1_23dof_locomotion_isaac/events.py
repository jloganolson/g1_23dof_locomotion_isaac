
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal
import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # return tensor
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_log_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    r"""Sample using log-uniform distribution within a range.

    The log-uniform distribution is defined as a uniform distribution in the log-space. It
    is useful for sampling values that span several orders of magnitude. The sampled values
    are uniformly distributed in the log-space and then exponentiated to get the final values.

    .. math::

        x = \exp(\text{uniform}(\log(\text{lower}), \log(\text{upper})))

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    # cast to tensor if not already
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=torch.float, device=device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=torch.float, device=device)
    # sample in log-space and exponentiate
    return torch.exp(sample_uniform(torch.log(lower), torch.log(upper), size, device))


def sample_gaussian(
    mean: torch.Tensor | float, std: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample using gaussian distribution.

    Args:
        mean: Mean of the gaussian.
        std: Std of the gaussian.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor.
    """
    if isinstance(mean, float):
        if isinstance(size, int):
            size = (size,)
        return torch.normal(mean=mean, std=std, size=size).to(device=device)
    else:
        return torch.normal(mean=mean, std=std).to(device=device)




def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = sample_uniform

    if distribution == "uniform":
        dist_fn = sample_uniform
    elif distribution == "log_uniform":
        dist_fn = sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn

def randomize_body_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()

    if not hasattr(env, "default_coms"):
        # Randomize robot base com
        env.default_coms = coms.clone()
        env.base_com_bias = torch.zeros((env.num_envs, 3), dtype=torch.float, device=coms.device)

    # apply randomization on default values
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_pd_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the scale of the pd gains by adding, scaling or setting random values.

    This function allows randomizing the scale of the pd gain. The function samples random values from the
    given distribution parameters and adds, or sets the values into the simulation based on the operation.

    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    kp_scale = env.default_kp_scale.clone()
    kd_scale = env.default_kd_scale.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        kp_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "abs":
        kp_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "scale":
        kp_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.kp_scale[env_ids, :] = kp_scale[env_ids, :]
    env.kd_scale[env_ids, :] = kd_scale[env_ids, :]



def randomize_action_noise_range(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the sample range of the added action noise by adding, scaling or setting random values.

    This function allows randomizing the scale of the sample range of the added action noise. The function
    samples random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    rfi_lim = env.default_rfi_lim.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        rfi_lim[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    elif operation == "abs":
        rfi_lim[env_ids, :] = dist_fn(*distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device)
    elif operation == "scale":
        rfi_lim[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.rfi_lim[env_ids, :] = rfi_lim[env_ids, :]
