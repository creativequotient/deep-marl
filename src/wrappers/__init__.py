from .base_wrapper import BaseWrapper
from .mpe_wrapper import MPEWrapper
from .sc2_wrapper import SC2Wrapper

REGISTRY = {
    'mpe.simple': MPEWrapper.make_mpe_env('simple', 0),
    'mpe.simple_adversary': MPEWrapper.make_mpe_env('simple_adversary', 1),
    'mpe.simple_push': MPEWrapper.make_mpe_env('simple_push', 1),
    'mpe.simple_speaker_listener': MPEWrapper.make_mpe_env('simple_speaker_listener', 2),
    'mpe.simple_spread': MPEWrapper.make_mpe_env('simple_spread', 3),
    'mpe.simple_tag': MPEWrapper.make_mpe_env('simple_tag', 3),
    'sc2.3m': SC2Wrapper(map_name='3m')
}
