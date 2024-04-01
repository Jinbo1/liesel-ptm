from pathlib import Path

import dill


def cache_filename(cache_dir: Path | str, func_name: str, args, kwargs) -> Path:
    cache_key = dill.dumps((func_name, args, kwargs))
    return Path(cache_dir) / f"{func_name}_{hash(cache_key)}.pkl"


def cache(directory):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            cache_dir = Path(directory)
            cache_dir.mkdir(exist_ok=True, parents=True)

            # Generate a unique filename based on function name and arguments
            cache_file = cache_filename(
                cache_dir, f"{func.__name__}_{id(func)}", args, kwargs
            )

            # If cached file exists, load and return the result
            if cache_file.exists():
                with open(cache_file, "rb") as file:
                    result = dill.load(file)
            else:
                # Execute the function and cache the result
                result = func(*args, **kwargs)
                with open(cache_file, "wb") as file:
                    dill.dump(result, file)

            return result

        return wrapper

    return decorator
