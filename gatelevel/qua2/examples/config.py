from dataclasses import dataclass, field
from typing import Tuple, Union, List, Set, Dict

from numpy import ndarray


@dataclass
class LogicalVertex:
    element_group: Union[str, int]
    element: Union[str, int]
    channel: Union[str, int]


@dataclass
class LogicalEdge:
    element_group: Union[str, int]
    element1: Union[str, int]
    element2: Union[str, int]
    channel: Union[str, int]


class WaveformPlaceholder:
    pass


@dataclass
class MixerCorrectionMatrix:
    c00 = 1.0
    c01 = 0.0
    c10 = 0.0
    c11 = 1.0


def make_default_correction():
    return [MixerCorrectionMatrix()]


@dataclass
class SingleWaveform:
    data: Union[WaveformPlaceholder, List[float]]
    name: str = field(default_factory=str)


@dataclass
class IQWaveform:
    data: Union[WaveformPlaceholder, List[complex]]
    name: str = field(default_factory=str)


@dataclass
class SingleOutputChannel:
    port: Tuple
    offset = 0.0
    logical_address: Union[LogicalEdge, LogicalVertex]
    waveforms: List[SingleWaveform] = field(default_factory=list)  # OK to not have waveforms, if we only want CW
    name: str = field(default_factory=str)


@dataclass
class IQOutputChannel:
    port_i: Tuple
    port_q: Tuple
    offset_i = 0.0
    offset_q = 0.0
    logical_address: Union[LogicalEdge, LogicalVertex]
    mixer_corrections: List[MixerCorrectionMatrix] = field(default_factory=make_default_correction())
    waveforms: List[IQWaveform] = field(default_factory=list)  # OK to not have waveforms, if we only want CW
    name: str = field(default_factory=str)


@dataclass
class SingleIntegrationWeight:
    data: List[complex]
    name: str = field(default_factory=str)


@dataclass
class IQIntegrationWeight:
    data: List[float]
    name: str = field(default_factory=str)


@dataclass
class SingleInputChannel:
    port: tuple
    offset = 0.0
    logical_address: Union[LogicalEdge, LogicalVertex]
    integration_weights: List[SingleIntegrationWeight] = field(default_factory=list)
    name: str = field(default_factory=str)


@dataclass
class IQInputChannel:
    port_i: tuple
    port_q: tuple
    offset_i = 0.0
    offset_q = 0.0
    logical_address: Union[LogicalEdge, LogicalVertex]
    mixer_corrections: List[MixerCorrectionMatrix] = field(default_factory=make_default_correction())
    integration_weights: List[IQIntegrationWeight] = field(default_factory=list)
    name: str = field(default_factory=str)


@dataclass
class QMConfig:
    channels: List[Union[SingleInputChannel, SingleOutputChannel, IQInputChannel, IQOutputChannel]]


def logical_vertex(group, element, channel):
    pass


def logical_edge(group, element1, element2, channel):
    pass
