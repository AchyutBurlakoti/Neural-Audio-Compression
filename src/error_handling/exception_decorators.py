from functools import wraps


class InvalidRaiseException(Exception):
    pass


def only_throws(E):
    """
    :source: https://stackoverflow.com/a/18289516
    """
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except E:
                raise
            except InvalidRaiseException:
                raise
            except Exception as e:
                raise InvalidRaiseException(
                    'got %s, expected %s, from %s' % (e.__class__.__name__, E.__name__, f.__name__)
                )

        return wrapped

    return decorator