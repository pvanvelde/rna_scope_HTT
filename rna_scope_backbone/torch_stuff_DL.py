
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple, Union, Optional
from torch import Tensor
import copy
import scipy



#######################################################################################
class VectorPSF_2D(torch.nn.Module):
    def __init__(self, VPSF):
        super().__init__()
        self.VPSF = VPSF


    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool =False):


        if pos is not None:
            x_ori = x*1
            x_new = torch.cat((pos,x_ori), dim=1)

        else:
            x_new = x*1

        x_new[:, 0:2] = (x_new[:, 0:2] - (self.VPSF.Mx / 2)) * self.VPSF.pixelsize
        x_new[:, [1, 0]] = x_new[:, [0, 1]]
        # insert z estim (0s)
        tensor_with_column = torch.cat((x_new[:, :2], torch.zeros_like(x_new[:,0])[...,None], x_new[:, 2:]), dim=1)
        if bg_only:

            tensor_with_column[:,3] = 0
            tensor_with_column = torch.cat((tensor_with_column, x_ori),
                                           dim=1)

        mu, jac = self.VPSF.poissonrate(tensor_with_column, bg_constant)  # needs to return mu and jac

        x_new[:, 0:2] = x_new[:, 0:2] / self.VPSF.pixelsize + self.VPSF.Mx / 2
        jac[:,:,:,[0,1]] =  jac[:,:,:,[1,0]]* self.VPSF.pixelsize
        # get rid of z
        jac = jac[:,:,:,[0,1,3,4]]
        if pos is not None:
            jac = jac[:,:,:,[2,3]]
        if bg_only == True:
            jac = jac[...,-1][...,None]
        #swap jac x and y
        return mu, jac

class LM_MLE_with_iter(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float, tol):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()
        self.tol = tol
        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)


        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, initial, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        dev = smp.device
        smp = torch.clamp(smp,1e-4).float()
        cur = initial * 1
        loglik = torch.empty(cur.size(dim=0), dtype=torch.float32)
        mu = torch.zeros(smp.size(), device=dev, dtype=torch.float32)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], cur.size()[1]), device=dev, dtype=torch.float32)

        scale = torch.zeros(cur.size(), device=dev, dtype=torch.float32)
        traces = torch.zeros((self.iterations + 1, cur.size()[0], cur.size()[1]), device=dev, dtype=torch.float32)
        traces[0, :, :] = cur

        tol = (torch.ones((cur.size()[0], cur.size()[1]), device=dev, dtype=torch.float32)
               * self.tol[None, ...].repeat([cur.size()[0], 1]).to(dev))
        good_array = torch.ones(cur.size()[0], device=dev, dtype=torch.bool)
        delta = torch.ones(cur.size(), device=dev, dtype=torch.float32)
        bool_array = torch.ones(cur.size(), device=dev, dtype=torch.bool)
        i = 0
        flag_tolerance = 0


        while (i < self.iterations) and (flag_tolerance == 0):
            if bg_constant is not None:
                bg_constant_temp=bg_constant[good_array, ...]
            else:
                bg_constant_temp = None
            if pos is not None:
                pos_temp = pos[good_array, ...]
            else:
                pos_temp = None
            temp1, temp2 =  self.model.forward(cur[good_array,...], bg_constant_temp, pos_temp,bg_only)
            mu[good_array,...]= temp1.float()
            jac[good_array, ...] = temp2.float()

            # cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
            #                                                      jac[good_array, :, :], smp[good_array, :], lambda_,
            #                                                      param_range_min_max, scale[good_array, :])
            cur[good_array, :], scale[good_array, :] = MLE_instead_lmupdate(cur[good_array, :], mu[good_array, :, :],
                                                                            jac[good_array, :, :, :],
                                                                            smp[good_array, :, :],
                                                                            self.lambda_, self.param_range_min_max)

            #mu, jac = self.model(cur, const_)
            #cur[good_array, ...], scale[good_array, ...] = lm_update(cur[good_array, ...], mu[good_array, ...]
                                #, jac[good_array, ...], smp[good_array, ...], self.lambda_, self.param_range_min_max, scale[good_array, ...])
            loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))

            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])
            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :])#.type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != cur.size(-1)
            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        return cur, loglik, traces

################################################################################
def compute_numerical_derivatives( theta, epsilon, deriv_analytical, numpixels, bg_constant):
    # Initialize numerical derivatives tensor
    deriv_numerical = torch.zeros_like(deriv_analytical)

    # Loop over parameters
    for i in range(5):
        # Add epsilon to parameter i
        theta_plus = theta.clone()
        theta_plus[:, i] += epsilon

        # Compute mu for theta_plus
        mu_plus, _ = gauss_psf_2D(theta_plus, numpixels, bg_constant)

        # Subtract epsilon from parameter i
        theta_minus = theta.clone()
        theta_minus[:, i] -= epsilon

        # Compute mu for theta_minus
        mu_minus, _ = gauss_psf_2D(theta_minus, numpixels, bg_constant)

        # Compute numerical derivative for parameter i
        deriv_numerical[..., i] = (mu_plus - mu_minus) / (2 * epsilon)

    return deriv_numerical
#@torch.jit.script
def gauss_psf_2D(theta, numpixels: int, bg_constant:Optional[torch.Tensor]=None):
    """
    theta: [x,y,N,bg,sigma].T
    """
    numpixels = torch.tensor(numpixels, device=theta.device)
    pi = 3.141592653589793  # torch needs to include pi
    pi = torch.tensor(pi, device=theta.device)
    x, y, N, bg, sigma = theta[:, 0]-0.5, theta[:, 1]-0.5, theta[:, 2], theta[:, 3], theta[:, 4]
    pixelpos = torch.arange(0, numpixels, device=theta.device)

    # Use the sqrt-based constants
    OneOverSqrt2PiSigma = (1.0 / (torch.sqrt(2 * pi) * sigma))[:, None, None]
    OneOverSqrt2Sigma = (1.0 / (torch.sqrt(torch.tensor(2.0, device=theta.device)) * sigma))[:, None, None]
    OneOverSqrt2PiSigmasqrd = (1.0 / (torch.sqrt(2 * pi) * sigma ** 2))[:, None, None]

    # Pixel centers
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]

    # Compute the erf arguments using the sqrt-based constant:
    Xexp0 = (Xc - x[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Xexp1 = (Xc - x[:, None, None] - 0.5) * OneOverSqrt2Sigma

    # For the derivative terms, use the corresponding constant:
    Xexp0sig = (Xc - x[:, None, None] + 0.5) * OneOverSqrt2PiSigmasqrd
    Xexp1sig = (Xc - x[:, None, None] - 0.5) * OneOverSqrt2PiSigmasqrd

    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigma * (torch.exp(-Xexp1 ** 2) - torch.exp(-Xexp0 ** 2))
    dEx_dSigma = -Xexp0sig * torch.exp(-Xexp0 ** 2) + Xexp1sig * torch.exp(-Xexp1 ** 2)

    Yexp0 = (Yc - y[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Yexp1 = (Yc - y[:, None, None] - 0.5) * OneOverSqrt2Sigma

    Yexp0sig = (Yc - y[:, None, None] + 0.5) * OneOverSqrt2PiSigmasqrd
    Yexp1sig = (Yc - y[:, None, None] - 0.5) * OneOverSqrt2PiSigmasqrd

    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigma * (torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))
    dEy_dSigma = -Yexp0sig * torch.exp(-Yexp0 ** 2) + Yexp1sig * torch.exp(-Yexp1 ** 2)
    if bg_constant is not None:
        mu = N[:, None, None] * Ex * Ey + bg_constant/torch.amax(bg_constant, dim=(-1, -2))[...,None,None] * bg[...,None,None]
        dmu_bg = bg_constant/torch.amax(bg_constant, dim=(-1, -2))[...,None,None]
    else:
        mu = N[:, None, None] * Ex * Ey + bg[:, None, None]
        dmu_bg = 1 + mu * 0
    dmu_x = N[:, None, None] * Ey * dEx
    dmu_y = N[:, None, None] * Ex * dEy
    dmu_I = Ex * Ey

    dmu_sigma = N[:, None, None] * (Ex * dEy_dSigma + dEx_dSigma * Ey)

    deriv = torch.stack((dmu_x, dmu_y, dmu_I, dmu_bg, dmu_sigma), -1)

    return mu, deriv


def show_tensor(img):
    import napari

    viewer, image_layer = napari.imshow(img.detach().cpu().numpy())
    napari.run()
@torch.jit.script
def gauss_psf_2D_fixed_sigma(theta, roisize: int, sigma: float,bg_constant:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    theta_ = torch.cat((theta, sigma_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., :-1]

@torch.jit.script
def gauss_psf_2D_fixed_sigma_fixed_pos(theta, roisize: int, sigma: float,bg_constant:Optional[torch.Tensor]=None,pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    theta_ = torch.cat((theta, sigma_), -1)
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((pos, theta_), -1)
    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 2:-1]



@torch.jit.script
def gauss_psf_2D_flex_sigma(theta, roisize: int, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    mu, jac = gauss_psf_2D(theta, roisize)
    return mu, jac[..., :]


#@torch.jit.script
def gauss_psf_2D_I_Bg(theta, roisize: int, sigma: float, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((theta, sigma_), -1)
    theta_ = torch.cat((pos, theta_), -1)
    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 2:-1]


@torch.jit.script
def gauss_psf_2D_Bg(theta, roisize: int, sigma: float, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    I = torch.zeros((len(theta), 1), device=theta.device)
    if pos is None:
        pos = torch.ones((len(theta), 2), device=theta.device) * roisize / 2
    theta_ = torch.cat((theta, sigma_), -1)
    theta_ = torch.cat((I, theta_), -1)
    theta_ = torch.cat((pos, theta_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize, bg_constant)
    return mu, jac[..., 3:-1]


class Gaussian2DFixedSigmaPSF(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_fixed_sigma(x, self.roisize, self.sigma, bg_constant)

class Gaussian2DFixedSigmaPSFFixedPos(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_fixed_sigma_fixed_pos(x, self.roisize, self.sigma, bg_constant, pos)

#@torch.jit.script
class Gaussian2D_IandBg(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_I_Bg(x, self.roisize, self.sigma, bg_constant, pos)

#@torch.jit.script
class Gaussian2D_Bg(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None , pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_Bg(x, self.roisize, self.sigma, bg_constant,pos)


class Gaussian_flexsigma(torch.nn.Module):
    def __init__(self, roisize):
        super().__init__()
        self.roisize = roisize

    def forward(self, x, bg_constant:Optional[torch.Tensor]=None , pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        return gauss_psf_2D_flex_sigma(x, self.roisize)


def compute_crlb(mu: Tensor, jac: Tensor, *, skip_axes: List[int] = []):
    """
    Compute crlb from expected value and per pixel derivatives.
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)

    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[..., axes]

    sample_dims = tuple(np.arange(1, len(mu.shape)))

    fisher = torch.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
    fisher = fisher / mu[..., None, None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = torch.zeros((len(mu), naxes), device=mu.device)
    crlb[:, axes] = torch.sqrt(torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2))
    return crlb
@torch.jit.script
def likelihood_v2(image, mu, dmudtheta):

    sample_dims = [-2, -1]
    sample_dimsjac = [-3, -2]
    varfit = 0
    # calculation of weight factors
    keps = 1e3 * 2.220446049250313e-16

    mupos = (mu > 0) * mu + (mu < 0) * keps

    weight = (image - mupos) / (mupos + varfit)
    dweight = (image + varfit) / (mupos + varfit) ** 2
    num_params = dmudtheta.size()[-1]

    # log-likelihood, gradient vector and Hessian matrix
    logL = torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), sample_dims)
    gradlogL = torch.sum(weight[..., None] * dmudtheta, sample_dimsjac)
    HessianlogL = torch.zeros((gradlogL.size(0), num_params, num_params))
    for ii in range(num_params):
        for jj in range(num_params):
            HessianlogL[:, ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj], sample_dims)

    return logL, gradlogL, HessianlogL

#@torch.jit.script
def MLE_instead_lmupdate(cur, mu, jac, smp, lambda_: float, param_range_min_max):
    """
    Separate some of the calculations to speed up with jit script
    """
    #
    merit, grad, Hessian = likelihood_v2(smp, mu, jac)
    grad = grad.float()
    diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
    b = torch.eye(diag.size(1))
    # hessian can be approximated by J^T*J
    c = diag.unsqueeze(2).expand(diag.size(0),diag.size(1),diag.size(1))

    diag_full = c * b
    # matty = Hessian + lambda_ * diag_full

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + lambda_ * diag_full
    Bmat = Bmat.to(cur.device)

    # try:
    #     dtheta = torch.linalg.solve(-Bmat, grad)
    # except:
    eye = torch.eye(Bmat.size(1), device=Bmat.device, dtype=Bmat.dtype)
    Bmat += eye * 1e-6
    Bmat_inv = torch.linalg.pinv(Bmat)  # shape (n, p, p)
    dtheta = -torch.matmul(Bmat_inv, grad.unsqueeze(-1)).squeeze(-1)  # (n, p)

    #dtheta = torch.linalg.solve(-Bmat, grad)
    dtheta = dtheta.float()
    dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
    cur = cur+ dtheta

    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))

    scale = 1
    return cur, scale

@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    # assert np.array_equal(smp.shape, mu.shape)
    sampledims = [i for i in range(1, len(smp.shape))]

    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp * invmu ** 2

    jacm = torch.matmul(jac[..., None], jac[..., None, :])
    alpha = jacm * af[..., None, None]
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp * invmu - 1)[..., None]).sum(sampledims)
    return alpha, beta


#@torch.jit.script
def lm_update(cur, mu, jac, smp, lambda_: float, param_range_min_max, scale_old=torch.Tensor(1)):
    """
    Separate some of the calculations to speed up with jit script
    """
    alpha, beta = lm_alphabeta(mu, jac, smp)
    scale_old = scale_old.to(device=cur.device)
    K = cur.shape[-1]

    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)
        scale = (alpha * alpha).sum(1)
        #scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent

        if scale_old.size() != torch.Size([1]):
            scale = torch.maximum(scale,scale_old)

        # assert torch.isnan(scale).sum()==0
        alpha += lambda_ * scale[:, :, None] * torch.eye(K, device=smp.device)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K, device=smp.device)[None]

    steps = torch.linalg.solve(alpha, beta)
    #steps[torch.isnan(steps)] = (cur[torch.isnan(steps)]+0.1)*0.9
    assert torch.isnan(cur).sum() == 0
    assert torch.isnan(steps).sum() == 0

    cur = cur + steps
    # if Tensor.dim(param_range_min_max) == 2:
    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))
    # elif Tensor.dim(param_range_min_max) == 3:
    #
    # cur = torch.maximum(cur, param_range_min_max[:, :, 0])
    # cur = torch.minimum(cur, param_range_min_max[:, :, 1])
    # else:
    #     raise 'check bounds'
    if scale_old.size() != torch.Size([1]):
        return cur, scale
    else:
        return cur, scale

class MLE_new(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model
    def forward(self, initial, smp, param_range_min_max,iterations:int, lambda_:float):
        """
            model:
                function that takes parameter array [batchsize, num_parameters]
                and returns (expected_value, jacobian).
                Expected value has shape [batchsize, sample dims...]
                Jacobian has shape [batchsize, sample dims...,num_parameters]

            initial: [batchsize, num_parameters]

            return value is a tuple with:
                estimates [batchsize,num_parameters]
                traces [iterations, batchsize, num_parameters]
        """
        dev = initial.dev
        # if not isinstance(initial, torch.Tensor):
        #     initial = torch.Tensor(initial).to(smp.device)
        cur = (initial * 1)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #     param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        traces = torch.zeros((iterations + 1, cur.size()[0], cur.size()[1]), device=dev)
        traces[0, :, :] = cur

        assert len(smp) == len(initial)
        scale = torch.zeros(cur.size(), device=dev)
        tol_ = torch.ones(cur.size()[1]).to(dev)*0.1
        mu = torch.zeros(smp.size()).to(dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],cur.size()[1])).to(dev)
        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tol_[None, ...].repeat([cur.size()[0], 1])
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(dev)
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)
        i = 0
        flag_tolerance = 0

        while (i < iterations) and (flag_tolerance == 0):
            mu[good_array, :], jac[good_array, :, :] = self.model.forward(cur[good_array, :])
            logL, gradlogL, HessianlogL = likelihood_v2(smp[good_array, :],mu[good_array, :], jac[good_array, :, :])
            # cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
            #                                                      jac[good_array, :, :], smp[good_array, :], lambda_,
            #                                                      param_range_min_max, scale[good_array, :])
            cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
                                                          gradlogL[good_array, :, :], smp[good_array, :], lambda_,
                                                          param_range_min_max, scale[good_array, :])
            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != cur.size()[1]

            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
        loglik[torch.isinf(loglik)]=1e-20
        return cur,loglik, traces

class Gaussian3DPSF(torch.nn.Module):
    def __init__(self, roisize, zslices, sigma=None, fit_bg_per_slice=True):
        super().__init__()
        self.sigma=sigma
        self.roisize = roisize
        self.zslices = zslices
        self.fit_bg_per_slice = fit_bg_per_slice
    def forward(self, x, bg_constant:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None, bg_only:bool=False):
        if self.fit_bg_per_slice:
            return gauss_psf_3D(x, self.roisize,  self.zslices, sigma=self.sigma, bg_constant=bg_constant )
        else:
            return gauss_psf_3D_const_bg_old(x, self.roisize,  self.zslices, sigma=self.sigma, bg_constant=bg_constant )


#@torch.jit.script
def gauss_psf_3D(theta, numpixels: int, zslices: int, bg_constant: Optional[torch.Tensor] = None, sigma=None):
    """
    Computes the 3D integrated Gaussian PSF with independent background for each z-slice.

    Parameterization:
      If sigma is None:
         theta: [x, y, z, N, bg_0, bg_1, ..., bg_{zslices-1}, sigma_x, sigma_y, sigma_z]
      Else:
         theta: [x, y, z, N, bg_0, bg_1, ..., bg_{zslices-1}]

    Note: The (x,y,z) positions are shifted by -0.5 to locate the emitter relative to pixel centers.
    """
    device = theta.device
    numpixels_t = torch.tensor(numpixels, device=device)
    zslices_t  = torch.tensor(zslices,  device=device)

    pi = torch.tensor(3.141592653589793, device=device)

    # ------------------------------------------------------------------
    # Extract parameters. Background is now a vector of length 'zslices'
    # ------------------------------------------------------------------
    if bg_constant is not None:
        if sigma is None:
            # Expected theta shape: [n, 4 + zslices + 3]
            x = theta[:, 0]
            y = theta[:, 1]
            z = theta[:, 2]
            N = theta[:, 3]
            bg = theta[:, 4:4+zslices]  # shape: [n, zslices]
            sigma_x = theta[:, 4+zslices]
            sigma_y = theta[:, 5+zslices]
            sigma_z = theta[:, 6+zslices]
        else:
            # Expected theta shape: [n, 4 + zslices]
            x = theta[:, 0]
            y = theta[:, 1]
            z = theta[:, 2]
            N = theta[:, 3]
            bg = theta[:, 4:4+zslices]  # shape: [n, zslices]
            sigma_x = torch.ones_like(x) * sigma[0]
            sigma_y = torch.ones_like(x) * sigma[1]
            sigma_z = torch.ones_like(x) * sigma[2]
    else:
        if sigma is None:
            # Expected theta shape: [n, 4 + zslices + 3]
            x = theta[:, 0]
            y = theta[:, 1]
            z = theta[:, 2]
            N = theta[:, 3]
            bg = theta[:, 4]  # shape: [n, zslices]
            sigma_x = theta[:, 5]
            sigma_y = theta[:, 6]
            sigma_z = theta[:, 7]
        else:
            # Expected theta shape: [n, 4 + zslices]
            x = theta[:, 0]
            y = theta[:, 1]
            z = theta[:, 2]
            N = theta[:, 3]
            bg = theta[:, 4]  # shape: [n, zslices]
            sigma_x = torch.ones_like(x) * sigma[0]
            sigma_y = torch.ones_like(x) * sigma[1]
            sigma_z = torch.ones_like(x) * sigma[2]

    # ---------------------------------------------------------------
    # Create pixel coordinate grids for x, y, and z dimensions.
    # ---------------------------------------------------------------
    pixelpos = torch.arange(numpixels_t, device=device)  # [0, 1, ..., numpixels-1]
    zpixelpos = torch.arange(zslices_t, device=device)     # [0, 1, ..., zslices-1]

    Xc = pixelpos[None, None, None, :]  # shape: (1, 1, 1, numpixels)
    Yc = pixelpos[None, None, :, None]  # shape: (1, 1, numpixels, 1)
    Zc = zpixelpos[None, :, None, None]  # shape: (1, zslices, 1, 1)

    # ------------------------------------------------------------------
    # Precompute Gaussian constants for each dimension.
    # ------------------------------------------------------------------
    # X dimension constants:
    OneOverSqrt2Sigmax    = (1.0 / (torch.sqrt(torch.tensor(2.0, device=device)) * sigma_x))[:, None, None, None]
    OneOverSqrt2PiSigmax  = (1.0 / (torch.sqrt(2.0 * pi) * sigma_x))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdx = (1.0 / (torch.sqrt(2.0 * pi) * sigma_x ** 2))[:, None, None, None]

    # Y dimension constants:
    OneOverSqrt2Sigmay    = (1.0 / (torch.sqrt(torch.tensor(2.0, device=device)) * sigma_y))[:, None, None, None]
    OneOverSqrt2PiSigmay  = (1.0 / (torch.sqrt(2.0 * pi) * sigma_y))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdy = (1.0 / (torch.sqrt(2.0 * pi) * sigma_y ** 2))[:, None, None, None]

    # Z dimension constants:
    OneOverSqrt2Sigmaz    = (1.0 / (torch.sqrt(torch.tensor(2.0, device=device)) * sigma_z))[:, None, None, None]
    OneOverSqrt2PiSigmaz  = (1.0 / (torch.sqrt(2.0 * pi) * sigma_z))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdz = (1.0 / (torch.sqrt(2.0 * pi) * sigma_z ** 2))[:, None, None, None]

    # ------------------------------------------------------------------
    # Compute the integrated Gaussian along each dimension.
    # ------------------------------------------------------------------
    # X-dimension:
    Xexp0 = (Xc - x[:, None, None, None] + 0.5) * OneOverSqrt2Sigmax
    Xexp1 = (Xc - x[:, None, None, None] - 0.5) * OneOverSqrt2Sigmax
    Xexp0sig = (Xc - x[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdx
    Xexp1sig = (Xc - x[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdx
    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigmax * (torch.exp(-Xexp1 ** 2) - torch.exp(-Xexp0 ** 2))
    dEx_dSigma = -Xexp0sig * torch.exp(-Xexp0 ** 2) + Xexp1sig * torch.exp(-Xexp1 ** 2)

    # Y-dimension:
    Yexp0 = (Yc - y[:, None, None, None] + 0.5) * OneOverSqrt2Sigmay
    Yexp1 = (Yc - y[:, None, None, None] - 0.5) * OneOverSqrt2Sigmay
    Yexp0sig = (Yc - y[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdy
    Yexp1sig = (Yc - y[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdy
    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigmay * (torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))
    dEy_dSigma = -Yexp0sig * torch.exp(-Yexp0 ** 2) + Yexp1sig * torch.exp(-Yexp1 ** 2)

    # Z-dimension:
    Zexp0 = (Zc - z[:, None, None, None] + 0.5) * OneOverSqrt2Sigmaz
    Zexp1 = (Zc - z[:, None, None, None] - 0.5) * OneOverSqrt2Sigmaz
    Zexp0sig = (Zc - z[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdz
    Zexp1sig = (Zc - z[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdz
    Ez = 0.5 * torch.erf(Zexp0) - 0.5 * torch.erf(Zexp1)
    dEz = OneOverSqrt2PiSigmaz * (torch.exp(-Zexp1 ** 2) - torch.exp(-Zexp0 ** 2))
    dEz_dSigma = -Zexp0sig * torch.exp(-Zexp0 ** 2) + Zexp1sig * torch.exp(-Zexp1 ** 2)

    # ------------------------------------------------------------------
    # Combine the integrated Gaussians.
    # ------------------------------------------------------------------
    # Note: Ex and Ey have shapes [n, 1, 1, numpixels] and [n, 1, numpixels, 1],
    # while Ez has shape [n, zslices, 1, 1] so that the product Ex*Ey*Ez is [n, zslices, numpixels, numpixels].
    if bg_constant is not None:
        # Normalize bg_constant (e.g., from a 2D estimation) and use it to scale the background for each slice.
        denom_bg = torch.amax(bg_constant, dim=(-1, -2, -3), keepdim=True)
        scale_bg = bg_constant / denom_bg  # expected shape broadcastable to [n, zslices, 1, 1]
        mu = N[:, None, None, None] * (Ex * Ey * Ez) + scale_bg * bg[:, :, None, None]
    else:
        mu = N[:, None, None, None] * (Ex * Ey * Ez) + bg[:, None, None,None]

    # ------------------------------------------------------------------
    # Compute derivatives with respect to each parameter.
    # ------------------------------------------------------------------
    # Derivatives for x, y, z, and N:
    dmu_x = N[:, None, None, None] * (dEx * Ey * Ez)
    dmu_y = N[:, None, None, None] * (Ex * dEy * Ez)
    dmu_z = N[:, None, None, None] * (Ex * Ey * dEz)
    dmu_N = Ex * Ey * Ez

    # ------------------------------------------------------------------
    # Derivative with respect to the background parameters:
    # Since each background parameter bg_i only affects slice i,
    # we create a list of derivative arrays—one per slice.
    # ------------------------------------------------------------------
    eye = torch.eye(zslices, device=mu.device).view(1, zslices, 1, 1, zslices)

    # Create a helper tensor to broadcast over the spatial dimensions.
    ones_spatial = torch.ones(mu.shape[0], 1, mu.shape[2], mu.shape[3], 1, device=mu.device)

    if bg_constant is not None:
        # Multiply the identity mask by scale_bg (expanded to match the shape) and by ones_spatial.
        dmu_bg = ones_spatial * scale_bg.unsqueeze(-1) * eye
        if sigma is None:
            dmu_sigma_x = N[:, None, None, None] * (dEx_dSigma * Ey * Ez)
            dmu_sigma_y = N[:, None, None, None] * (Ex * dEy_dSigma * Ez)
            dmu_sigma_z = N[:, None, None, None] * (Ex * Ey * dEz_dSigma)
            # Stack all derivatives: parameters order: [x, y, z, N, bg_0, ..., bg_{zslices-1}, sigma_x, sigma_y, sigma_z]
            deriv = torch.concatenate(
                [dmu_x[..., None], dmu_y[..., None], dmu_z[..., None], dmu_N[..., None]] + [dmu_bg] + [
                    dmu_sigma_x[..., None], dmu_sigma_y[..., None], dmu_sigma_z[..., None]],
                dim=-1
            )
        else:
            # When sigma is provided, parameters order: [x, y, z, N, bg_0, ..., bg_{zslices-1}]
            deriv = torch.concatenate([dmu_x[..., None], dmu_y[..., None], dmu_z[..., None], dmu_N[..., None], dmu_bg],
                                      dim=-1)
    else:
        # If no scaling constant, simply use the identity mask expanded to the full shape.
        dmu_bg = ones_spatial * eye
        dmu_bg = torch.ones_like(dmu_N)
        if sigma is None:
            dmu_sigma_x = N[:, None, None, None] * (dEx_dSigma * Ey * Ez)
            dmu_sigma_y = N[:, None, None, None] * (Ex * dEy_dSigma * Ez)
            dmu_sigma_z = N[:, None, None, None] * (Ex * Ey * dEz_dSigma)
            # Stack all derivatives: parameters order: [x, y, z, N, bg_0, ..., bg_{zslices-1}, sigma_x, sigma_y, sigma_z]
            deriv = torch.concatenate(
                [dmu_x[..., None], dmu_y[..., None], dmu_z[..., None], dmu_N[..., None], dmu_bg[..., None]]  + [
                    dmu_sigma_x[..., None], dmu_sigma_y[..., None], dmu_sigma_z[..., None]],
                dim=-1
            )
        else:
            # When sigma is provided, parameters order: [x, y, z, N, bg_0, ..., bg_{zslices-1}]
            deriv = torch.concatenate([dmu_x[..., None], dmu_y[..., None], dmu_z[..., None], dmu_N[..., None], dmu_bg[..., None]],
                                      dim=-1)
    # ------------------------------------------------------------------
    # If sigma is not provided, also compute derivatives with respect to sigma.
    # ------------------------------------------------------------------


    return mu.float(), deriv.float()



def gauss_psf_3D_const_bg(theta, numpixels: int, zslices: int , bg_constant: Optional[torch.Tensor] = None, sigma=None):
    """
    theta: [x, y, z, N, bg, sigma_x, sigma_y, sigma_z].T
    """
    #numpixels = torch.tensor(numpixels, device=theta.device)
    device = theta.device
    numpixels_t = torch.tensor(numpixels, device=device)
    zslices_t  = torch.tensor(zslices,  device=device)


    pi = 3.141592653589793  # torch needs to include pi
    pi = torch.tensor(pi, device=theta.device)
    if sigma == None:
        x, y, z, N, bg, sigma_x, sigma_y, sigma_z = theta[:, 0] - 0.5, theta[:, 1] - 0.5, theta[:, 2]- 0.5, theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6], theta[:, 7]
    else:
        x, y, z, N, bg  = theta[:, 0] - 0.5, theta[:, 1] - 0.5, theta[:, 2]- 0.5, theta[:, 3], theta[:, 4]
        sigma_x = torch.ones_like(x)*sigma[0]
        sigma_y = torch.ones_like(x)*sigma[1]
        sigma_z= torch.ones_like(x)*sigma[2]
        # Create pixel coordinates
    pixelpos = torch.arange(numpixels_t, device=device)  # [0..numpixels-1]
    zpixelpos = torch.arange(zslices_t, device=device)  # [0..zslices-1]

    # Constants, matching the 2D style
    pi = torch.tensor(3.141592653589793, device=device)

    # ---------------------------------------------------------------
    #  Define constants for X dimension analogous to z:
    # ---------------------------------------------------------------
    OneOverSqrt2Sigmax = (1.0 / (torch.sqrt(torch.tensor(2.0, device=device)) * sigma_x))[:, None, None, None]
    OneOverSqrt2PiSigmax = (1.0 / (torch.sqrt(2.0 * pi) * sigma_x))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdx = (1.0 / (torch.sqrt(2.0 * pi) * sigma_x ** 2))[:, None, None, None]

    # ---------------------------------------------------------------
    #  Define constants for Y dimension analogous to z:
    # ---------------------------------------------------------------
    OneOverSqrt2Sigmay = (1.0 / (torch.sqrt(torch.tensor(2.0, device=device)) * sigma_y))[:, None, None, None]
    OneOverSqrt2PiSigmay = (1.0 / (torch.sqrt(2.0 * pi) * sigma_y))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdy = (1.0 / (torch.sqrt(2.0 * pi) * sigma_y ** 2))[:, None, None, None]

    # ---------------------------------------------------------------
    #  Z dimension constants remain as before:
    # ---------------------------------------------------------------
    OneOverSqrt2PiSigmaz = (1.0 / (torch.sqrt(2 * pi) * sigma_z))[:, None, None, None]
    OneOverSqrt2Sigmaz = (1.0 / (torch.sqrt(torch.tensor(2.0, device=theta.device)) * sigma_z))[:, None, None, None]
    OneOverSqrt2PiSigmasqrdz = (1.0 / (torch.sqrt(2 * pi) * sigma_z ** 2))[:, None, None, None]

    # Build the "pixel-center" grid:
    Xc = pixelpos[None, None, None, :]  # shape (1, 1, 1, numpixels)
    Yc = pixelpos[None, None, :, None]  # shape (1, 1, numpixels, 1)
    Zc = zpixelpos[None, :, None, None]  # shape (1, zslices, 1, 1)

    # ---------------------------------------------------------------
    #  Compute E_x (the integrated Gaussian along x):
    # ---------------------------------------------------------------
    Xexp0 = (Xc - x[:, None, None, None] + 0.5) * OneOverSqrt2Sigmax
    Xexp1 = (Xc - x[:, None, None, None] - 0.5) * OneOverSqrt2Sigmax

    Xexp0sig = (Xc - x[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdx
    Xexp1sig = (Xc - x[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdx

    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigmax * (torch.exp(-Xexp1 ** 2) - torch.exp(-Xexp0 ** 2))
    dEx_dSigma = -Xexp0sig * torch.exp(-Xexp0 ** 2) + Xexp1sig * torch.exp(-Xexp1 ** 2)

    # ---------------------------------------------------------------
    #  Compute E_y (the integrated Gaussian along y):
    # ---------------------------------------------------------------
    Yexp0 = (Yc - y[:, None, None, None] + 0.5) * OneOverSqrt2Sigmay
    Yexp1 = (Yc - y[:, None, None, None] - 0.5) * OneOverSqrt2Sigmay

    Yexp0sig = (Yc - y[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdy
    Yexp1sig = (Yc - y[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdy

    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigmay * (torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))
    dEy_dSigma = -Yexp0sig * torch.exp(-Yexp0 ** 2) + Yexp1sig * torch.exp(-Yexp1 ** 2)

    # ---------------------------------------------------------------
    #  Compute E_z (as before):
    # ---------------------------------------------------------------
    Zexp0 = (Zc - z[:, None, None, None] + 0.5) * OneOverSqrt2Sigmaz
    Zexp1 = (Zc - z[:, None, None, None] - 0.5) * OneOverSqrt2Sigmaz

    Zexp0sig = (Zc - z[:, None, None, None] + 0.5) * OneOverSqrt2PiSigmasqrdz
    Zexp1sig = (Zc - z[:, None, None, None] - 0.5) * OneOverSqrt2PiSigmasqrdz

    Ez = 0.5 * torch.erf(Zexp0) - 0.5 * torch.erf(Zexp1)
    dEz = OneOverSqrt2PiSigmaz * (torch.exp(-Zexp1 ** 2) - torch.exp(-Zexp0 ** 2))
    dEz_dSigma = -Zexp0sig * torch.exp(-Zexp0 ** 2) + Zexp1sig * torch.exp(-Zexp1 ** 2)
    # ---------------------------------------------------------------
    #  Combine E_x, E_y, E_z -> mu
    # ---------------------------------------------------------------
    #   mu = N * (Ex * Ey * Ez) + background-term
    if bg_constant is not None:
        # Typically you’d normalize it as in your 2D code:
        denom_bg = torch.amax(bg_constant, dim=(-1, -2, -3), keepdim=True)
        scale_bg = bg_constant / denom_bg
        mu = N[:, None, None, None] * (Ex * Ey * Ez) + scale_bg * bg[:, None, None, None]
        dmu_bg = scale_bg
    else:
        mu = N[:, None, None, None] * (Ex * Ey * Ez) + bg[:, None, None, None]
        dmu_bg = 1.0 + mu * 0.0  # derivative w.r.t. bg is 1

    # ---------------------------------------------------------------
    #  Derivatives w.r.t. each parameter
    # ---------------------------------------------------------------
    #  dmu/dx:
    dmu_x = N[:, None, None, None] * (dEx * Ey * Ez)
    #  dmu/dy:
    dmu_y = N[:, None, None, None] * (Ex * dEy * Ez)
    #  dmu/dz:
    dmu_z = N[:, None, None, None] * (Ex * Ey * dEz)

    #  dmu/dN:
    dmu_N = Ex * Ey * Ez
    #  dmu/d(bg) = dmu_bg (already shaped correctly)

    #  dmu/dsigma_x:
    #    *only the x-part changes w.r.t sigma_x*
    #    => we treat the full product Ex*Ey*Ez as (dEx/dsigma_x)*Ey*Ez
    #    The other terms (Ex, Ey, Ez) do NOT have sigma_x in them if sigma_x != sigma_y,z
    dmu_sigma_x = N[:, None, None, None] * (dEx_dSigma * Ey * Ez)

    #  dmu/dsigma_y:
    dmu_sigma_y = N[:, None, None, None] * (Ex * dEy_dSigma * Ez)

    #  dmu/dsigma_z:
    dmu_sigma_z = N[:, None, None, None] * (Ex * Ey * dEz_dSigma)

    if sigma == None:
        deriv = torch.stack((dmu_x, dmu_y, dmu_z, dmu_N, dmu_bg, dmu_sigma_x, dmu_sigma_y, dmu_sigma_z), -1)
    else:
        deriv = torch.stack((dmu_x, dmu_y, dmu_z, dmu_N, dmu_bg), -1)
    return mu.float(), deriv.float()


def LM_MLE_for_zstack(model, smp_ori, min_max, initial_guess,tollim,dev,Nitermax,
                      damping_lm=0.001,roisize=16 ):
    smp_test = smp_ori.view(-1, smp_ori.size(-2), smp_ori.size(-1))
    zslices = smp_ori.shape[1]
    if roisize < smp_ori.size(-1):
        # Calculate the difference and offsets for slicing
        diff = smp_ori.size(-1) - roisize
        offset = diff // 2  # Calculate offset for centering

        # Handle both odd and even sizes
        start = offset
        end = offset + roisize

        # Slice the center of the ROI
        smp = smp_ori[..., start:end, start:end]
    else:
        smp = smp_ori
        start = 0
        end = smp_ori.size(-1)
        offset=0
    # median over the last 3 dims (jointly)

    # or with quantile (works on your version; single dim only)
    bg = torch.quantile(smp.flatten(-3, -1), 0.5, dim=-1)
    initial_guess[:,4] = torch.clamp(bg, min=0.1, max=1e30)

    # show_tensor(torch.concatenate((smp_test,bg_estim),dim=-1))

    # Estimate the emitter intensity N by summing over the signal (smp) after background subtraction
    # across all slices and spatial dimensions.
    initial_guess[:, 3] = (torch.sum(smp, dim=(-1, -2, -3)) - (bg*smp.size(-1)*smp.size(-2)*smp.size(-3)))
    initial_guess[:, 3] = torch.clamp(initial_guess[:, 3], min=0.1,max=1e30)
    initial_guess[:, 3] = torch.clamp(initial_guess[:, 3], min=0.1, max=1e30)
    test = initial_guess.detach().cpu().numpy()

    pfa_around_pos_l = []
    cur = initial_guess * 1
    loglik = torch.zeros(smp.size()[0]).to(dev)
    mu = torch.zeros(smp.size()).to(dev).to(torch.float32)

    jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], smp.size()[3], cur.size()[1])).to(dev).to(torch.float32)


    traces = torch.zeros((Nitermax + 1, cur.size()[0], cur.size()[1])).to(dev).to(torch.float32)
    traces[0, :, :] = cur

    tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tollim
    good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
    delta = torch.ones(cur.size()).to(dev)
    bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)

    i = 0
    flag_tolerance = 0
    while (i < Nitermax) and (flag_tolerance == 0):


        mu[good_array, ...], jac[good_array, ...] = model.forward(cur[good_array, :],
                                                                          bg_constant=None)
        #show_tensor(torch.concatenate((smp, mu), dim=-1))
        cur[good_array, ...],loglik[good_array, ...] = MLE_update_forzstack(cur[good_array, ...], mu[good_array, ...],
                                                         jac[good_array, ...],
                                                         smp[good_array, ...], min_max, damping_lm=damping_lm, )

        cur_2d = cur.repeat_interleave(smp.size(1), dim=0)

        traces[i + 1, good_array, ...] = cur[good_array, ...]
        delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :]) / traces[i,
                                                                                                         good_array,
                                                                                                         :]

        bool_array[good_array] = (delta[good_array, ...] < tol[good_array, ...]).type(torch.bool)
        test = torch.sum(bool_array, dim=1)
        good_array = test !=  cur.size()[1]  # two for photons and bg
        #show_tensor(torch.concatenate((smp, mu,bg_estim3D_resized), dim=-1))
        if torch.sum(good_array) == 0:
            flag_tolerance = 1
        i = i + 1
    ### glrt stuff

    if model.sigma is None:
        pass
    else:

        cur_2d[..., [0, 1]] += offset
        # show_tensor(torch.concatenate((smp_test,bg_estim),dim=-1))
        ratio_all, _, _, mu_iandbg, _, traces_bg, traces_i = \
            glrtfunction(smp_test, mu.size(0)*mu.size(1)+1, min_max[[0, 1, 3, 4]], cur_2d[..., [0, 1, 3, 4]], smp_test.size(-1),
                         model.sigma[0], tol=torch.Tensor([1e-3, 1e-3]).to(dev), lambda_=1e-4,
                         iterations=30, bg_constant=None, vector=False, use_pos=True)

        def normcdf(x):
            return 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))

        def fast_harmonic(n):
            """Returns an approximate value of n-th harmonic number.
               http://en.wikipedia.org/wiki/Harmonic_number
            """
            # Euler-Mascheroni constant
            gamma = 0.57721566490153286060651209008240243104215933593992
            return gamma + np.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

            # ratio = 2 * (loglik_int_all - loglik_bg_all)
            # ratio = np.reshape(ratio, ( (np.size(track_rnp, 0)),
            #     (np.size(track_rnp, 1) - self.roisize), (np.size(track_rnp, 2) - self.roisize)))

        N = len(ratio_all)

        original_array = np.arange(0, N)

        pfa = 2 * normcdf(-np.sqrt(np.clip(ratio_all.detach().cpu().numpy(), 0, np.inf)))
        ratio_test = ratio_all.detach().cpu().numpy()

        # show_tensor(smp_test)
        argsort = np.argsort(pfa)
        aargsort = np.argsort(argsort)
        pfa_sorted = np.sort(pfa)
        numtests = len(pfa_sorted)
        c_m = fast_harmonic(numtests)
        alfa_thresh = 0.05
        thres_arr = np.arange(1, numtests + 1, 1) / (numtests * c_m) * alfa_thresh
        # thres_arr = np.arange(1, numtests + 1, 1) / (numtests ) * alfa
        good_arr_sorted = pfa_sorted <= thres_arr
        pfa_sorted_adj = numtests * c_m / np.arange(1, numtests + 1, 1) * pfa_sorted

        pfa_adj = pfa_sorted_adj[aargsort]  # undo sort
        good_arr = good_arr_sorted[aargsort]

        num_items = len(pfa_adj)
        trimmed_length = (num_items // 10) * 10  # Trim to nearest multiple of 10
        trimmed_pfa_test = pfa_adj[:trimmed_length]

        # Reshape into groups of 10 and compute the median
        pfa_test_reshaped = trimmed_pfa_test.reshape(-1, 10)
        z_pos_arr = cur[:, 2].detach().cpu().numpy()

        pfa_around_pos_l = []
        slice_radius = 2.5  # We'll end up with 5 slices total around the center

        for z, z_pos in enumerate(z_pos_arr):
            # Round z_pos to the nearest integer (center index)
            center_idx = int(round(z_pos))

            # How many indices to take on each side of the center
            half_span = int(np.floor(slice_radius))  # For 2.5, this is 2

            # Define the start and end (exclusive) for slicing
            start_idx = center_idx - half_span
            end_idx = center_idx + half_span + 1  # +1 because Python slice end is exclusive

            pfa_around_pos = pfa_test_reshaped[z, start_idx:end_idx]

            # We expect an odd number of elements: 2*half_span + 1
            expected_length = 2 * half_span + 1  # For slice_radius=2.5 → 5

            # If the slice is out of bounds or otherwise truncated, fill with ones
            if len(pfa_around_pos) != expected_length:
                pfa_around_pos = np.ones(expected_length)

            pfa_around_pos_l.append(pfa_around_pos)
        #print('iteration: ', i, ' out of ', Nitermax)

    return cur, traces, pfa_around_pos_l, None
def MLE_update_forzstack( cur, mu, jac, smp, param_range_min_max,damping_lm=0.001):
    def likelihood(image, mu, dmudtheta):
        numparams = dmudtheta.shape[-1]

        keps = 1e3 * torch.finfo(mu.dtype).eps
        keps = torch.tensor(keps, dtype=torch.float32).to(image.device)  # Replace ... with your actual tensor
        mupos = torch.where(mu > 0, mu, keps)
        varfit = 0
        weight = (image - mupos) / (mupos + varfit)
        dweight = (image + varfit) / (mupos + varfit) ** 2


        sampledim = (1, 2, 3)

        logL = torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), dim=sampledim)
        gradlogL = torch.sum(weight[..., None] * dmudtheta, dim=sampledim)
        # if self.zstack:
        #     HessianlogL = torch.zeros((numparams, numparams))
        #
        #     for ii in range(numparams):
        #         for jj in range(numparams):
        #             HessianlogL[ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj])
        # else:
        HessianlogL = torch.zeros((gradlogL.size(0), numparams, numparams))

        for ii in range(numparams):
            for jj in range(numparams):
                HessianlogL[:, ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj], sampledim)

        return logL, gradlogL, HessianlogL

    """
    Separate some of the calculations to speed up with jit script
    """

    sample_dim = (1, 2, 3, 4)

    merit, grad, Hessian = likelihood(smp, mu, jac)
    # torch.sum(torch.isnan(Hessian))
    diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
    b = torch.eye(diag.size(1))
    c = diag.unsqueeze(2).expand(diag.size(0), diag.size(1), diag.size(1))
    diag_full = c * b
    # matty = Hessian + lambda_ * diag_full

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + damping_lm * diag_full
    epsilon = 1e-6
    identity = torch.eye(Bmat.size(-1), device=Bmat.device).unsqueeze(0).expand(Bmat.size(0), -1, -1)
    Bmat +=   epsilon * identity

    Bmat = Bmat.to(cur.device)
    grad = grad.to(cur.device)

    Bmat_inv = torch.linalg.pinv(Bmat)  # shape (n, p, p)
    dtheta = -torch.matmul(Bmat_inv, grad.unsqueeze(-1)).squeeze(-1)  # (n, p)

    #dtheta = torch.linalg.solve(-Bmat, grad)
    dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
    cur = cur + dtheta

    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))

    return cur, merit

def glrtfunction(smp_arr, batch_size:int, bounds, initial_arr, roisize:int,sigma:float, tol, lambda_:float=1e-5, iterations:int=30,
                 bg_constant:Optional[torch.Tensor]=None,use_pos:Optional[bool]=False, vector:Optional[bool]=False, GT=False):
    n_iterations = smp_arr.size(0) // batch_size + int(smp_arr.size(0) % batch_size > 0)

    loglik_bg_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    loglik_int_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    traces_bg_all = torch.zeros((smp_arr.size(0),iterations+1,1), device=smp_arr.device)
    traces_int_all = torch.zeros((smp_arr.size(0),iterations+1,2), device=smp_arr.device)
    mu_iandbg_list = []


    modelIbg = Gaussian2D_IandBg(roisize, sigma)
    modelbg  = Gaussian2D_Bg(roisize, sigma)
    mle_Ibg = LM_MLE_with_iter(modelIbg, lambda_=lambda_, iterations=iterations,
                           param_range_min_max=bounds[[2, 3], :], tol=tol)
    bg_params = bounds[3, :]
    bg_params = bg_params[None, ...]
    mle_bg = LM_MLE_with_iter(modelbg, lambda_=lambda_, iterations=iterations, param_range_min_max=bg_params,
                           tol=tol[:1])
    for batch in range(n_iterations):
        smp_ = smp_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :, :]
        initial_ = initial_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :]
        if bg_constant is not None:
            bg_constant_batch = bg_constant[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :, :]
            if not GT:
                std_dev = bg_constant_batch.std(dim=(-2, -1))

                # Calculate the mean value along the last two axes
                mean_val = bg_constant_batch.mean(dim=(-2, -1), keepdim=True).expand_as(bg_constant_batch)
                # Create a mask where the standard deviation is less than 2
                mask = (std_dev < 4).unsqueeze(-1).unsqueeze(-1).expand_as(bg_constant_batch)
                # Use the mask to replace slices with their mean value where the condition is met
                bg_constant_batch = torch.where(mask, mean_val, bg_constant_batch)
        else:
            bg_constant_batch = None

        if use_pos:
            pos = initial_[:, :2]
        else:
            pos = None
        with torch.no_grad():  # when no tensor.backward() is used

            # setup model and compute Likelhood for hypothesis I and Bg



            # mle = LM_MLE(model, lambda_=1e-3, iterations=40,
            #              param_range_min_max=param_range[[2, 3], :], traces=True)
            if vector:
                #mle = torch.compile(mle)
                test = 0  # select if single gpus
            else:
                #pass
                mle_Ibg = torch.jit.script(mle_Ibg)  # select if single gpus

            if vector == True and pos == None:
                pos_in = torch.ones_like(initial_[:, [2, 3]])*roisize/2
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in,
                                                                 bg_only=False)
                mu_iandbg, _ = modelIbg.forward(params_, bg_constant_batch, pos_in)
            elif vector == True:
                pos_in = copy.copy(pos)
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in, bg_only=False)
                mu_iandbg, _ = modelIbg.forward(params_, bg_constant_batch, pos_in)
            else:
                pos_in = copy.copy(pos)
                params_, loglik_I_andbg, traces_iandbg = mle_Ibg.forward(smp_, initial_[:, [2, 3]], bg_constant_batch, pos_in)
                mu_iandbg, _ = gauss_psf_2D_I_Bg(params_, roisize, sigma, bg_constant_batch, pos_in)
            mu_iandbg_list.append(mu_iandbg)



            bg = initial_[:, 3]
            bg = bg[..., None]




            if vector:
                #mle = torch.compile(mle)
                test = 0  # select if single gpus
            else:
                #pass
                mle_bg = torch.jit.script(mle_bg)


            # setup model and compute Likelhood for hypothesis Bg
            if vector == True and pos == None:
                pos_in = torch.ones_like(initial_[:, [2, 3]]) * roisize / 2
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_, bg, bg_constant_batch, pos_in,
                                                                     bg_only=True)
                mu_bg, _ = model.forward(params_, bg_constant_batch, pos_in,     bg_only=True)
            elif vector == True:
                pos_in = copy.copy(pos)
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_, bg, bg_constant_batch, pos_in,
                                                                     bg_only=True)
                mu_bg, _ = model.forward(params_, bg_constant_batch, pos_in, bg_only=True)
            else:
                pos_in = copy.copy(pos)
                params_bg_, loglik_bgonly, traces_bgonly = mle_bg.forward(smp_[:, :, :], bg, bg_constant_batch, pos_in)
                mu_bg, _ = gauss_psf_2D_Bg(params_bg_, roisize, sigma, bg_constant_batch, pos)







            loglik_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_bgonly))] = loglik_bgonly
            loglik_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg))] = loglik_I_andbg
            traces_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_bgonly,[1,0,2])
            traces_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_iandbg,[1,0,2])

    ratio = 2 * (loglik_int_all - loglik_bg_all)
    #ratio[torch.isnan(ratio)] = 0
    return ratio, loglik_int_all, loglik_bg_all, mu_iandbg_list,mu_bg, traces_bg_all, traces_int_all
