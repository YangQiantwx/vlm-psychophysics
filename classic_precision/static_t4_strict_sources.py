from .static_t4_sources import CONTRACT, INDICES, descriptor, digest, read, require, verified, write_new
from . import static_t4_sources as parent

NAME = 'static-t4-strict-first-last'

def protocol(original):
    return parent.protocol(original) | dict(experiment=NAME, source_aliases=768, bias_only_aliases=0,
        maximum_new_model_calls=3072, source_master='FFV1 level 3 gbrp16le sRGB',
        native_model_input='RGB8 native X image', parent_protocol='classic/t4/strict-v1')
