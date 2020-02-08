def component_state(model_state_dict, prefix):
    component_state_dict = {key: value for key, value in model_state_dict.items() if key.startswith(prefix)}
    return component_state_dict