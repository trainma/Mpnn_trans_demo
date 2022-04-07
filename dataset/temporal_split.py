from typing import Union, Tuple

from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_graph_static_signal import DynamicGraphStaticSignal

from torch_geometric_temporal.signal.static_graph_temporal_signal_batch import StaticGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_graph_temporal_signal_batch import DynamicGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_graph_static_signal_batch import DynamicGraphStaticSignalBatch

Discrete_Signal = Union[
    StaticGraphTemporalSignal,
    StaticGraphTemporalSignalBatch,
    DynamicGraphTemporalSignal,
    DynamicGraphTemporalSignalBatch,
    DynamicGraphStaticSignal,
    DynamicGraphStaticSignalBatch,
]


def temporal_signal_split_valid(
        data_iterator, ratio: list
) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """
    assert len(ratio) == 3 and ratio[0] + ratio[1] + ratio[2] == 1
    train_ratio = ratio[0]
    valid_ratio = ratio[1]
    test_ratio = ratio[2]
    train_snapshots = int(train_ratio * data_iterator.snapshot_count)
    valid_snapshots = int((valid_ratio + train_ratio) * data_iterator.snapshot_count)

    if type(data_iterator) == StaticGraphTemporalSignal:
        train_iterator = StaticGraphTemporalSignal(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            data_iterator.features[0:train_snapshots],
            data_iterator.targets[0:train_snapshots],
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        valid_iterator = StaticGraphTemporalSignal(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            data_iterator.features[train_snapshots:valid_snapshots],
            data_iterator.targets[train_snapshots:valid_snapshots],
            **{key: getattr(data_iterator, key)[train_snapshots:valid_snapshots] for key in
               data_iterator.additional_feature_keys}
        )

        test_iterator = StaticGraphTemporalSignal(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            data_iterator.features[valid_snapshots:],
            data_iterator.targets[valid_snapshots:],
            **{key: getattr(data_iterator, key)[valid_snapshots:] for key in data_iterator.additional_feature_keys}
        )

    elif type(data_iterator) == DynamicGraphTemporalSignal:
        train_iterator = DynamicGraphTemporalSignal(
            data_iterator.edge_indices[0:train_snapshots],
            data_iterator.edge_weights[0:train_snapshots],
            data_iterator.features[0:train_snapshots],
            data_iterator.targets[0:train_snapshots],
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = DynamicGraphTemporalSignal(
            data_iterator.edge_indices[train_snapshots:],
            data_iterator.edge_weights[train_snapshots:],
            data_iterator.features[train_snapshots:],
            data_iterator.targets[train_snapshots:],
            **{key: getattr(data_iterator, key)[train_snapshots:] for key in data_iterator.additional_feature_keys}
        )

    elif type(data_iterator) == DynamicGraphStaticSignal:
        train_iterator = DynamicGraphStaticSignal(
            data_iterator.edge_indices[0:train_snapshots],
            data_iterator.edge_weights[0:train_snapshots],
            data_iterator.feature,
            data_iterator.targets[0:train_snapshots],
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = DynamicGraphStaticSignal(
            data_iterator.edge_indices[train_snapshots:],
            data_iterator.edge_weights[train_snapshots:],
            data_iterator.feature,
            data_iterator.targets[train_snapshots:],
            **{key: getattr(data_iterator, key)[train_snapshots:] for key in data_iterator.additional_feature_keys}
        )

    if type(data_iterator) == StaticGraphTemporalSignalBatch:
        train_iterator = StaticGraphTemporalSignalBatch(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            data_iterator.features[0:train_snapshots],
            data_iterator.targets[0:train_snapshots],
            data_iterator.batches,
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = StaticGraphTemporalSignalBatch(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            data_iterator.features[train_snapshots:],
            data_iterator.targets[train_snapshots:],
            data_iterator.batches,
            **{key: getattr(data_iterator, key)[train_snapshots:] for key in data_iterator.additional_feature_keys}
        )

    elif type(data_iterator) == DynamicGraphTemporalSignalBatch:
        train_iterator = DynamicGraphTemporalSignalBatch(
            data_iterator.edge_indices[0:train_snapshots],
            data_iterator.edge_weights[0:train_snapshots],
            data_iterator.features[0:train_snapshots],
            data_iterator.targets[0:train_snapshots],
            data_iterator.batches[0:train_snapshots],
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = DynamicGraphTemporalSignalBatch(
            data_iterator.edge_indices[train_snapshots:],
            data_iterator.edge_weights[train_snapshots:],
            data_iterator.features[train_snapshots:],
            data_iterator.targets[train_snapshots:],
            data_iterator.batches[train_snapshots:],
            **{key: getattr(data_iterator, key)[train_snapshots:] for key in data_iterator.additional_feature_keys}
        )

    elif type(data_iterator) == DynamicGraphStaticSignalBatch:
        train_iterator = DynamicGraphStaticSignalBatch(
            data_iterator.edge_indices[0:train_snapshots],
            data_iterator.edge_weights[0:train_snapshots],
            data_iterator.feature,
            data_iterator.targets[0:train_snapshots],
            data_iterator.batches[0:train_snapshots:],
            **{key: getattr(data_iterator, key)[0:train_snapshots] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = DynamicGraphStaticSignalBatch(
            data_iterator.edge_indices[train_snapshots:],
            data_iterator.edge_weights[train_snapshots:],
            data_iterator.feature,
            data_iterator.targets[train_snapshots:],
            data_iterator.batches[train_snapshots:],
            **{key: getattr(data_iterator, key)[train_snapshots:] for key in data_iterator.additional_feature_keys}
        )
    return train_iterator, valid_iterator, test_iterator
