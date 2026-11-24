# Clean Source Releases

The public deliverable is source, configurations, documentation and synthetic
CPU tests only. Historical experiment logs and deployment-specific records do
not belong in the release. Existing research evidence should be retained in a
separate private archive, not deleted or rewritten as a new validation result.

## Build And Inspect

1. Review `.release-files`. New source files are excluded until explicitly added.
2. Run `python -m pytest -q` on the intended source snapshot.
3. Run `python scripts/build_release.py` for the text/privacy audit.
4. Run `python scripts/build_release.py --output dist/source` in a fresh output directory.
5. Inspect `tar -tzf dist/source/vita-reference-source.tar.gz` and `SHA256SUMS`.
6. Build wheel/sdist only from an extraction of this clean source archive.

The builder reads only allowlisted text files. It rejects disallowed paths,
symlinks, binaries, selected secret formats, private home/mount paths, emails
and notebook host identifiers. Diagnostics name only the file, line and rule,
never the matched secret. Archive membership/content is rechecked before
writing. UID/GID, user/group names, timestamps and gzip file names are
normalized; `.git` and its history are not included.

`.gitignore`, `.gitattributes` and `MANIFEST.in` add defense in depth, but they do
not erase git history and do not replace the explicit release audit. Do not
publish the entire research folder or rely on `git add .` as sanitization.
Automated scanning cannot guarantee detection of every personal detail or
credential format; manual review of new prose and allowlist changes is required.

## Never Include Automatically

- Runtime manifests, command logs, failure traces and run receipts.
- Model/optimizer checkpoints, foundation weights, datasets and videos.
- Environment files, SSH aliases/configurations, keys, credentials or tokens.
- Personal home paths, remote mount paths, hostnames, account IDs and emails.
- Prior git history, private archives or machine-generated cache files.

New training runs may record absolute paths for reproducibility. Even exported
weights can contain model asset paths or data contracts. Publish those only
through a separately reviewed data/model release, with appropriate upstream
licenses and privacy checks. The source builder deliberately does not include
any trained or pretrained model.

## Licensing

Original project code is supplied under MIT. The paper, LIBERO, simulator and
foundation dependencies are not relicensed. Dataset/model access and redistribution
conditions must be reviewed separately. No author affiliation or endorsement is
claimed. No public hosting service is contacted by the release script.
