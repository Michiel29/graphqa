def select_component_state(model_state_dict, prefix):
    """Returns new state dict with only model parameters starting with prefix"""
    component_state_dict = {key: value for key, value in model_state_dict.items() if key.startswith(prefix)}
    return component_state_dict