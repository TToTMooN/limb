Run ruff linter and formatter on the limb package.

Steps:
1. Check for lint errors:
   ```bash
   uv run ruff check limb/
   ```

2. Auto-fix safe issues:
   ```bash
   uv run ruff check --fix limb/
   ```

3. Format code:
   ```bash
   uv run ruff format limb/
   ```

Line length is 119 chars. Config is in pyproject.toml [tool.ruff].
Do not modify pyproject.toml ruff settings unless the user asks.
Report any remaining issues that need manual fixes.
