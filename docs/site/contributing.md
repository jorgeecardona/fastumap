# Contributing

--8<-- "README.md:development"

## The docs site

This site is built with [MkDocs](https://www.mkdocs.org/) and Material, from `docs/site/`:

```bash
uv sync --extra docs
uv run mkdocs serve             # local preview at 127.0.0.1:8000
uv run mkdocs build --strict    # exactly what CI runs before publishing
```

Every prose page pulls its text from `README.md` with pymdownx snippets
(`--8<-- "README.md:section"`), so the README is the single source of truth and the site cannot
drift from it. To add a page, mark a new section in the README with
`<!-- --8<-- [start:name] -->` / `<!-- --8<-- [end:name] -->`, pull it into a page, and add that
page to `nav` in `mkdocs.yml`.
