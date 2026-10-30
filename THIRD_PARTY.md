# Third-party materials

`phase_ssm/vendor/bdh.py` is derived from [Pathway BDH](https://github.com/pathwaycom/bdh), commit `a1ea64154600d863b116f9a23e15a096e052e151`.
Its Python syntax tree matches the upstream file at that revision; the local file adds provenance and the complete MIT notice.
Copyright 2025 Pathway Technology, Inc.; the retained notice applies to that file.

The rest of this source release uses the MIT licence in `LICENSE`.
Python dependencies retain their own licences and are installed separately.

The training loader downloads [text8](http://mattmahoney.net/dc/textdata.html) into the ignored `data/` directory when requested.
The corpus is not redistributed in this repository; its source and applicable terms are separate from the code licence.
No trained checkpoints, model-derived weights, teacher transcripts, downloaded corpora, or generated topology bundles are included.
