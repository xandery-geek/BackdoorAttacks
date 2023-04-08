# modified from [TrojanZoo](https://github.com/ain-soph/trojanzoo)


from utils.output import ansi, prints


class BasicObject:
    r"""A basic class with a pretty :meth:`summary()` method.

    Attributes:
        name (str): The name of the instance or class.
        param_list (dict[str, list[str]]): Map from category strings to variable name list.
        indent (int): The indent when calling :meth:`summary()`. Defaults to ``0``.
    """
    name: str = 'basic_object'

    def __init__(self, indent: int = 0, **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['verbose'] = ['indent']
        self.indent = indent

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = None):
        r"""Summary the variables of the instance
        according to :attr:`param_list`.

        Args:
            indent (int): The space indent for the entire string.
                Defaults to :attr:`self.indent`.

        See Also:
            :meth:`trojanzoo.models.Model.summary()`.
        """
        indent = indent if indent is not None else self.indent
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi), indent=indent)
        prints('{yellow}{0}{reset}'.format(self.__class__.__name__, **ansi), indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(
                    key, **ansi), indent=indent + 10)
                prints({v: str(getattr(self, v)).split('\n')[0]
                       for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)

    def __str__(self) -> str:
        self.summary()