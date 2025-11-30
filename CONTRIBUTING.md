# Guía de Contribución

## Flujo de trabajo
1. Crea una rama desde `main`: `feature/<nombre>`
2. Commits pequeños y claros (puedes usar Conventional Commits).
3. Abre un Pull Request a `main` y pide revisión.
4. La CI debe pasar antes de fusionar.

## Estilo (elige lo que aplique)
- **Node**: eslint + prettier
- **Python**: ruff/flake8 + black + pytest

## Tests
Añade tests en `tests/` siempre que sea posible.
