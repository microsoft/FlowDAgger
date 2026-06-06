"""Argument parsing helpers: parse_training_args and AttrDict."""


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


def parse_training_args(train_args_dict, parser):
    for k, v in train_args_dict.items():
        if type(v) == tuple:
            parser.add_argument('--' + k, nargs="+", default=v, type=type(v[0]))
        elif type(v) != bool:
            parser.add_argument('--' + k, default=v, type=type(v))
        else:
            parser.add_argument('--' + k, default=int(v), type=int)
    args = parser.parse_args()
    config = {}
    for key in train_args_dict.keys():
        config[key] = getattr(args, key)
    variant = AttrDict(vars(args))
    variant['train_kwargs'] = config
    return variant, args
