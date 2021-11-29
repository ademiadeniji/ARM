import torch 

def make_optimizer(model, cfg, return_params=False, additional_params=[]):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for name, m in model.named_modules():

        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif name in skip or (
                (len(p.shape) == 1 or name.endswith(".bias"))
                and cfg.OPTIM.ZERO_WD_1D_PARAM
            ):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)
    non_bn_parameters += additional_params
    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        {"params": zero_parameters, "weight_decay": 0.0},
    ]
    optim_params = [x for x in optim_params if len(x["params"])]  
  

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == \
        len(non_bn_parameters) + len(bn_parameters) + len(zero_parameters) + len(no_grad_parameters) \
        - len(additional_params), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(list(model.parameters())),
    )
    print(
        "Making optimizer for embedding net params, counted: bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )
    method = cfg.OPTIM.METHOD
    if method == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIM.BASE_LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV,
        )
    elif method == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIM.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        ) 

    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(method)
        )
    
    if return_params:
        return optimizer, optim_params
         
    return optimizer