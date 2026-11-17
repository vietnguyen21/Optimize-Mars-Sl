"""Position-addressed embedding sampling; no training RNG is consumed."""

import hashlib


SAMPLING_STRATEGIES = ('sequential', 'epoch_shuffle_v1')


class EmbeddingSampler:
    def __init__(self, size, seed=42, strategy='sequential'):
        if size < 1 or strategy not in SAMPLING_STRATEGIES:
            raise ValueError('Positive example count and supported sampling strategy required')
        self.size, self.seed, self.strategy = size, seed, strategy
        self._epoch = None
        self._order = None

    def index(self, position):
        if position < 0:
            raise ValueError('Consumed-example position must be nonnegative')
        epoch, offset = divmod(position, self.size)
        if self.strategy == 'sequential':
            return offset
        if epoch != self._epoch:
            # v1: SHA-256 rank of each index, with index as collision tie-break.
            # Fully specified across Python versions; cache affects speed only.
            prefix = f'epoch_shuffle_v1:{self.seed}:{epoch}:'.encode('ascii')
            self._order = sorted(range(self.size), key=lambda i: (
                hashlib.sha256(prefix + str(i).encode('ascii')).digest(), i))
            self._epoch = epoch
        return self._order[offset]


def validate_embedding_extra(saved, expected):
    """Only pre-sampler contracts may omit strategy, meaning sequential v1."""
    contract = dict(saved.get('contract', {}))
    if 'sampling_strategy' not in contract and 'sampling_version' not in contract:
        contract.update(sampling_strategy='sequential', sampling_version=1)
    if contract != expected['contract']:
        raise ValueError('Embedding resume data/config/sampling differ from saved checkpoint')
