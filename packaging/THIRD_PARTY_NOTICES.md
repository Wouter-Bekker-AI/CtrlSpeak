# Differential-update tool

CtrlSpeak includes an unmodified Xdelta 3.2.2 executable by Joshua MacDonald,
copyright (C) Joshua MacDonald, under the Apache License, Version 2.0.

Source and release: https://github.com/jmacd/xdelta/tree/v3.2.2

The corresponding upstream license is bundled as `tools/xdelta3/LICENSE`.
Builds stage the original archives and license from
`packaging/vendor/xdelta3/` and verify the archive, executable and license
against SHA-256 pins in `utils/delta_tools.py`. CtrlSpeak's release signature
then covers the application containing this tool. No tool is fetched from
the internet during the build's tool-provisioning step or an installed update.

Other bundled components retain their respective upstream licenses.
