"""Main BaSiC class."""

from pydantic import BaseModel, Field, PrivateAttr, model_validator
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import dask.array as da
import torch

from skimage.transform import resize as skimage_resize
import torch.nn.functional as F
import time
import torch_dct as dct
import copy


# initialize logger with the package name
logger = logging.getLogger(__name__)


class BaSiC(BaseModel):
    """A class for fitting and applying BaSiC illumination correction profiles."""

    baseline: Optional[np.ndarray] = Field(
        None,
        description="Holds the baseline for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    darkfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the darkfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    fitting_mode: str = Field(
        "approximate", description="Must be one of ['ladmap', 'approximate']"
    )
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    flatfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the flatfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    smoothness_flatfield: float = Field(
        None, description="Weight of the flatfield term in the Lagrangian."
    )
    smoothness_darkfield: float = Field(
        None, description="Weight of the darkfield term in the Lagrangian."
    )
    sparse_cost_darkfield: float = Field(
        0.01, description="Weight of the darkfield sparse term in the Lagrangian."
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    max_reweight_iterations: int = Field(
        10,
        description="Maximum number of reweighting iterations.",
    )
    max_reweight_iterations_baseline: int = Field(
        5,
        description="Maximum number of reweighting iterations for baseline.",
    )
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-3,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-2,
        description="Optimization tolerance for update diff.",
    )
    resize_params: Dict = Field(
        {},
        description="Parameters for the resize function when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Whether or not to sort the intensities of the image.",
    )
    working_size: Optional[Union[int, List[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )
    device: str = Field(
        None,
        description="Must be one of ['cpu', 'cuda']",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _weight_dark: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)
    _smoothness_flatfield: float = PrivateAttr(None)
    _smoothness_darkfield: float = PrivateAttr(None)
    _sparse_cost_darkfield: float = PrivateAttr(None)
    _flatfield_small: float = PrivateAttr(None)
    _darkfield_small: float = PrivateAttr(None)
    _converge_flag: bool = PrivateAttr(None)

    class Config:
        """Pydantic class configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    @model_validator(mode="before")
    def debug_log_values(cls, values: Dict[str, Any]):
        """Use a validator to echo input values."""
        logger.debug("Initializing BaSiC with parameters:")
        for k, v in values.items():
            logger.debug(f"{k}: {v}")
        return values

    def __call__(
        self,
        images: Union[np.ndarray, da.core.Array, torch.Tensor],
        timelapse: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for `BaSiC.fit_transform`."""
        return self.fit_transform(images, timelapse)

    def _resize(self, Im, target_shape):
        if isinstance(Im, np.ndarray):
            Im2 = skimage_resize(
                Im,
                [1, 1] + target_shape,
                preserve_range=True,
                order=1,
                mode="edge",
                anti_aliasing=True,
            )

        elif isinstance(Im, da.core.Array):
            assert np.array_equal(target_shape[:-2], Im.shape[:-2])
            Im2 = (
                da.from_array(
                    [
                        skimage_resize(
                            np.array(Im[tuple(inds)]),
                            target_shape[-2:],
                            preserve_range=True,
                            **self.resize_params,
                        )
                        for inds in np.ndindex(Im.shape[:-2])
                    ]
                )
                .reshape((*Im.shape[:-2], *target_shape[-2:]))
                .compute()
            )

        elif isinstance(Im, torch.Tensor):
            Im2 = F.interpolate(
                Im,
                target_shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        else:
            raise ValueError(
                "Input must be either numpy.ndarray, dask.core.Array, or torch.Tensor."
            )
        return Im2

    def _resize_to_working_size(self, Im):
        """Resize the images to the working size."""
        if np.isscalar(self.working_size):
            working_shape = [self.working_size] * (Im.ndim - 2)
        else:
            if not Im.ndim - 2 == len(self.working_size):
                raise ValueError(
                    "working_size must be a scalar or match the image dimensions"
                )
            else:
                working_shape = self.working_size
        target_shape = [*Im.shape[:2], *working_shape]
        Im = self._resize(Im, target_shape)

        return Im

    def fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
        for_autotune=False,
    ) -> None:
        """Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional or 4-dimensional array
                    with dimension of (T,Y,X) or (T,Z,Y,X).
                    T can be either of time or mosaic position.
                    Multichannel images should be
                    independently corrected for each channel.
            fitting_weight: Relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy import datasets as bdata
            >>> images = bdata.wsi_brain()
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, None, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, None, ...]
        elif images.ndim == 4:
            if self.fitting_mode == "approximate":
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )

        if images.shape[-1] < 10 and not skip_shape_warning:
            logger.warning(
                "Image last dimension is less than 10. "
                + "Are you supplying images with the channel dimension?"
                + "Multichannel images should be "
                + "independently corrected for each channel."
            )

        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()

        Im = self._resize_to_working_size(images)

        if self.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(Im, np.ndarray):
            Im = torch.from_numpy(Im.astype(np.float32)).to(self.device)
        else:
            Im = Im.to(self.device)

        if fitting_weight is not None:
            flag_segmentation = True
            Ws = self._resize_to_working_size(fitting_weight) > 0
            if isinstance(Ws, np.ndarray):
                Ws = torch.from_numpy(Ws).to(self.device)
            else:
                Ws = Ws.to(self.device)
        else:
            flag_segmentation = False
            Ws = torch.ones_like(Im)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = torch.argsort(Im, dim=0)
            Im2 = torch.take_along_dim(Im, inds, dim=0)
            Ws2 = torch.take_along_dim(Ws, inds, dim=0)
        else:
            Im2 = Im
            Ws2 = Ws

        if self.smoothness_flatfield is None:
            meanD = Im.mean(0)
            meanD = meanD / meanD.mean()
            W_meanD = dct.dct_2d(meanD, norm="ortho")
            self._smoothness_flatfield = torch.sum(torch.abs(W_meanD)) / (400) * 0.5
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
        if self.smoothness_darkfield is None:
            self._smoothness_darkfield = self._smoothness_flatfield * 0.1
        else:
            self._smoothness_darkfield = self.smoothness_darkfield
        if self.sparse_cost_darkfield is None:
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        logger.debug(f"_smoothness_flatfield set to {self._smoothness_flatfield}")
        logger.debug(f"_smoothness_darkfield set to {self._smoothness_darkfield}")
        logger.debug(f"_sparse_cost_darkfield set to {self._sparse_cost_darkfield}")

        _temp = torch.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp[1][0]

        if self.fitting_mode == "approximate":
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im2.shape)
        fit_params = self.model_dump()
        fit_params.update(
            dict(
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=torch.min(Im2),
                image_norm=torch.linalg.norm(Im2),
            )
        )

        # Initialize variables
        W = torch.ones_like(Im2, dtype=torch.float32) * Ws2
        if flag_segmentation:
            W[Ws2 == 0] = self.epsilon
        W = W * W.numel() / W.sum()
        W_D = torch.zeros(
            Im2.shape[1:],
            dtype=torch.float32,
            device=self.device,
        )
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == "ladmap":
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        for i in range(self.max_reweight_iterations):
            logger.debug(f"reweighting iteration {i}")
            if self.fitting_mode == "approximate":
                S = torch.ones(Im2.shape[1:], dtype=torch.float32, device=self.device)
            else:
                S = torch.median(Im2, dim=0)
            S_hat = dct.dct_2d(S)
            D_R = torch.zeros(Im2.shape[1:], dtype=torch.float32, device=self.device)
            D_Z = 0.0
            if self.fitting_mode == "approximate":
                B = copy.deepcopy(Im2)
                B[Ws2 == 0] = torch.nan
                B = torch.squeeze(torch.nanmean(B, dim=(-2, -1))) / torch.nanmean(B)
                B = torch.nan_to_num(B)
            else:
                B = torch.ones(Im2.shape[0], dtype=torch.float32, device=self.device)

            I_R = torch.zeros(Im2.shape, dtype=torch.float32, device=self.device)
            I_B = (S * B[:, None, None])[:, None, ...] + D_R[None, ...]
