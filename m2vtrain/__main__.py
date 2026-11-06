"""`python -m m2vtrain ...`"""

from m2vtrain.cli import main

if __name__ == "__main__":  # worker processes for several teacher GPUs import this module without running it
    raise SystemExit(main())
