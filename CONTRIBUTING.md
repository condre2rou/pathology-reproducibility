# Contributing

Use Python 3.12 and the pinned dependencies. Start with `reproduce.py example`
and the tests. Add changes to a new branch after the repository is initialised.

Preserve seeds, analysis units, sampling rules and task membership unless a
scientific change is intended. Keep new results separate from `reference/` and
document changes to estimates. Test-data fixtures are for code correctness;
tutorials must use documented real public data.

Do not commit hospital records, feature vectors, image paths, reports, model
weights, access tokens, cached downloads or runtime workspaces. Public features
remain subject to upstream terms. Original project code uses the [MIT License](LICENSE);
preserve applicable copyright and license notices. See [LICENSING.md](LICENSING.md)
for the scope of the code license and separate third-party terms.

When reporting a bug, include the command, Python/package versions and a
sanitised error message. Do not attach private clinical inputs or unreviewed logs.
