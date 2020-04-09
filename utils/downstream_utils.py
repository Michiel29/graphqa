import numpy as np

def load_downstream_data(samples, trainer):
    features, targets = None, None
    
    for batch in samples:
        if batch is None:
            continue
        batch = trainer._prepare_sample(batch)
        batch_features, batch_targets = trainer.model(batch)
        if features is None:
            features, targets = batch_features, batch_targets
        else:
            features = np.concatenate((features, batch_features), axis=0)
            targets = np.concatenate((targets, batch_targets), axis=0)

    return features, targets