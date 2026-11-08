"""Immutable dataset identity and validated manifest paths."""

from pathlib import PurePosixPath


REPO_ID = "DrQY/vlm-psychophysics"


REVISION = "8a63496a8c59d72c1a0cb28f56c4b57325694f27"


SCOPES = tuple(
    f"{domain}/t{task}" for domain in ("classic", "natural") for task in range(1, 6)
)


PAIRED_SCOPES = ("classic/t5/paired-v1", "natural/t5/paired-v1")


def payload_path(scope, name):
    path = PurePosixPath(name)
    if (
        "\\" in name
        or path.as_posix() != name
        or path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(PurePosixPath(scope))
        or path == PurePosixPath(scope)
        or any(char in name for char in "*?[]")
    ):
        raise ValueError(f"Unexpected {scope} release path: {name}")
    return name
