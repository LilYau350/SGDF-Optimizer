import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    """
    Quintic Newton–Schulz iteration to compute a "zeroth power" / orthogonalization-like transform.

    Notes:
    - Works best on GPU with bfloat16 support (as in the reference Muon code).
    - For CPU or devices without bf16 support, it will still run but may be slower / less stable.
    """
    assert G.ndim >= 2

    a, b, c = (3.4445, -4.7750, 2.0315)

    # Prefer bf16 on supported devices; otherwise fallback to original dtype.
    try:
        X = G.to(dtype=torch.bfloat16)
    except Exception:
        X = G

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Ensure spectral norm is at most 1 (simple normalization)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.mT

    # Return in original dtype (match update dtype)
    return X.to(dtype=G.dtype)


def muon_postprocess(update: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """
    Apply Muon-style orthogonalization post-processing to an update tensor.
    - If update is conv-like (ndim==4), reshape to [out, -1], process, then reshape back.
    - If update is matrix-like (ndim>=2), process directly.
    - If update is vector/scalar (ndim<2), return unchanged.
    """
    if update.ndim < 2:
        return update

    if update.ndim == 4:
        u2 = update.view(update.size(0), -1)
        u2 = zeropower_via_newtonschulz5(u2, steps=ns_steps)
        # keep the same scaling trick as reference Muon code
        u2 = u2 * (max(1.0, update.size(-2) / update.size(-1)) ** 0.5)
        return u2.view_as(update)

    u2 = zeropower_via_newtonschulz5(update, steps=ns_steps)
    u2 = u2 * (max(1.0, update.size(-2) / update.size(-1)) ** 0.5)
    return u2


class SGDF(Optimizer):
    """
    Param-group options (all optional):
      - use_muon (bool): apply Muon postprocess to update for ndim>=2 params
      - ns_steps (int): Newton–Schulz steps for Muon postprocess (default 5, must be >=2 if use_muon)
      - use_sign (bool): apply sign() to update
        Note: if use_muon=True and param.ndim>=2, Muon takes precedence over sign.
    """
    def __init__(self,params, lr=0.5, betas=(0.9, 0.999), eps=1e-8, gamma=0.5, weight_decay=0.0, 
                            weight_decouple=False, use_sign=False, use_muon=False, ns_steps=5,):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid beta value: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if gamma <= 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if use_muon and ns_steps < 2:
            raise ValueError(f"When use_muon=True, ns_steps ({ns_steps}) should be >= 2")

        # IMPORTANT: defaults must reflect __init__ args (so you can set them outside)
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, weight_decay=weight_decay,
                        weight_decouple=weight_decouple, use_sign=use_sign, use_muon=use_muon,ns_steps=ns_steps,)
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            gamma = group["gamma"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decouple = bool(group.get("weight_decouple", False))
            use_muon = bool(group.get("use_muon", False))
            ns_steps = int(group.get("ns_steps", 5))
            use_sign = bool(group.get("use_sign", False))

            if use_muon and ns_steps < 2:
                raise ValueError(f"When use_muon=True, ns_steps ({ns_steps}) should be >= 2")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if weight_decay != 0.0:
                    # Apply weight decay, optional decoupling
                    if weight_decouple:
                        p.mul_(1 - lr * weight_decay)          # Decoupled weight decay
                    else:
                        grad.add_(p, alpha=weight_decay)       # Regular weight decay

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_var"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_var = state["exp_var"]

                # Compute gradient 1st and 2nd 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                grad_residual = (grad - exp_avg)
                
                exp_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                state['step'] += 1

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = (1 + beta1) * (1 - beta2**state['step']) / ((1 - beta1) * (1 - beta1**(2*state['step'])))
                
                exp_avg_corr = exp_avg / bias_correction1
                exp_var_corr = exp_var / bias_correction2

                # Estimation gain
                grad_hat_residual = grad - exp_avg_corr                 
                denom = grad_hat_residual.square()             
                denom.add_(exp_var_corr).add_(eps) 
                K = exp_var_corr.div_(denom)
                
                # apply gamma in-place on K
                if gamma == 1.0:
                    pass
                elif gamma == 0.5:
                    K.sqrt_()
                else:
                    K.pow_(gamma)
    
                # Gradient estimation
                update = exp_avg_corr + K * grad_hat_residual

                # -------- muon / sign --------
                if use_muon and update.ndim >= 2:
                    update = muon_postprocess(update, ns_steps=ns_steps)
                elif use_sign:
                    update = update.sign_()

                p.add_(update, alpha=-lr)

        return loss
