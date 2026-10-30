"""Allow ``python3 -m argus ...`` without any global PATH setup."""

from argus.cli import app

if __name__ == "__main__":
    app(prog_name="argus")
