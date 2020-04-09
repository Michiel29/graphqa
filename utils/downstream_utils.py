import numpy as np
from sklearn.preprocessing import StandardScaler

def load_downstream_data(samples, trainer, scaler=None):
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

    if scaler == 'standard':
        s = StandardScaler()
        features = s.fit_transform(features)

    return features, targets