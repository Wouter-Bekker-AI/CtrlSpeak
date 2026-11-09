# Pinned Xdelta build inputs

These are the original upstream Xdelta 3.2.2 archives, not CtrlSpeak release
executables. They are checked in so a clean build does not depend on the
upstream release CDN. The binaries installed by CtrlSpeak are unchanged.

- [Upstream release and source](https://github.com/jmacd/xdelta/releases/tag/v3.2.2)
- [Upstream license](https://github.com/jmacd/xdelta/blob/v3.2.2/xdelta3/LICENSE)

`scripts/fetch_patch_tool.py` stages the archive member and upstream
license into the ignored build cache. Archive, binary and license SHA-256
pins are in `utils/delta_tools.py`. The archive is never extracted wholesale
or executed; the one pinned executable is extracted by exact member name
and verified before any use.
