"""Run the real training CLI with a runtime test.csv read barrier."""
import builtins
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_bp


if __name__ == '__main__':
    original = builtins.open
    attempted = []

    def guarded(path, *args, **kwargs):
        if str(path).replace('\\', '/').endswith('/test.csv'):
            attempted.append(str(path))
            raise AssertionError('Training attempted to open test.csv')
        return original(path, *args, **kwargs)

    with patch('builtins.open', guarded):
        code = train_bp.main(sys.argv[1:])
    print(f'Runtime test.csv read attempts: {len(attempted)}', flush=True)
    sys.exit(code or bool(attempted))
