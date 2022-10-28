

def resolve_data_config(args, model=None):

    if model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg
        if default_cfg["crop_pct"] is not None:
            args.crop_pct = default_cfg["crop_pct"]
    
    return args
