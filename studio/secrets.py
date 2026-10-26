"""Keys live in ~/studio-data/secrets as KEY=VALUE files only their owner can read, never in the repository."""

from pathlib import Path


class SecretError(RuntimeError):
    pass


def read_secret(env_file: Path, name: str) -> str:
    env_file = Path(env_file)
    if not env_file.exists():
        raise SecretError(f"{env_file} is missing; put {name}=... in it with mode 0600")
    if env_file.stat().st_mode & 0o077:
        raise SecretError(f"{env_file} must be mode 0600")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().removeprefix("export ").partition("=")
        if key.strip() == name and value.strip():
            return value.strip().strip("'\"")
    raise SecretError(f"no {name} in {env_file}")
