def clear_cuda():
    import torch
    import gc
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            pass

# Little utility function for peeking at model stats
def print_trainable_parameters(model) -> str:
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"

def print_model_size(model):
    """Prints the size of the model in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return "model size: {:.3f}MB".format(size_all_mb)

def get_model_dtype(model):
    """Returns the number of parameters in the model, grouped by dtype."""
    params: dict[str, int] = {}
    for param in model.parameters():
        dtype = str(param.dtype)
        if dtype not in params:
            params[dtype] = 0
        params[dtype] += param.nelement()

    for param in model.buffers():
        dtype = str(param.dtype)
        if dtype not in params:
            params[dtype] = 0
        params[dtype] += param.nelement()

    return params
