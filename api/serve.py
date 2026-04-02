"""Entry point for the 'serve' console script."""

import uvicorn


def main() -> None:
    """Run the API server via uvicorn."""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
