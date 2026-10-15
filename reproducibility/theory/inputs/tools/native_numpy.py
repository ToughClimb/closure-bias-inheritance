"""Read saved native MLP arrays; selected methods copied from revision/reliability_core.py."""
import json
import numpy as np
from scipy.special import expit
class NumpyMLPClosure:
    def __init__(self, metadata, arrays):
            self.metadata = metadata
            self.arrays = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
            if metadata['backbone'] != 'mlp':
                raise ValueError('Native export currently supports the polynomial-residual MLP')
            if not all(np.isfinite(value).all() for value in self.arrays.values()):
                raise FloatingPointError('Nonfinite native network export')
    def scalar(self, branch, values):
            values = np.asarray(values, dtype=np.float64)
            specification = self.metadata['branches'][branch]
            span = max(specification['upper'] - specification['lower'], 1e-6)
            scaled = 2 * (values - specification['lower']) / span - 1
            polynomial = np.polynomial.polynomial.polyval(scaled, self.arrays[branch + '_coefficients'])
            hidden = scaled[..., None]
            for index in range(len(specification['layers'])):
                hidden = hidden @ self.arrays[f'{branch}_weight_{index}'].T + self.arrays[f'{branch}_bias_{index}']
                if index + 1 < len(specification['layers']):
                    hidden = np.tanh(hidden)
            return polynomial + specification['residual_scale'] * hidden[..., 0]
    def diffusion(self, values):
            return self.metadata['min_diffusion'] + np.exp(self.arrays['log_diffusion_scale']) * expit(self.scalar('D', values))
    def reaction(self, values):
            return self.scalar('R', values)
    @classmethod
    def load(cls, path):
            with np.load(path, allow_pickle=False) as values:
                metadata = json.loads(str(values['metadata']))
                arrays = {key: values[key].copy() for key in values.files if key != 'metadata'}
            return cls(metadata, arrays)
