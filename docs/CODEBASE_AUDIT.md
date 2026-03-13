# Codebase Audit: Duplicates and Code Smells

Automated scan across tracked code/docs-like text files.
- Tracked files: 1777
- Candidate code/docs files: 1315
- Text files analyzed: 1286

## High-priority bug fixes applied in this pass

- Fixed wrong delegation: `remove_byAddress(...)` in `embeddings/main.cpp` called `delete_byAddress(...)`; now it correctly calls `remove_byAddress(...)`.
- Fixed `merge(...)` behavior in `embeddings/src/BertIndexManager.cpp` to execute full `merge()` instead of `merge_last_two()`.
- Fixed bad prototype typo in disabled production append overload (`int64_end` -> `int64_t end`) in `embeddings/main.cpp`.

## Exact duplicate files (byte-for-byte)

1. (4 files)
   - `lib/python2.7/IB.2.20211006.4.0a.py`
   - `lib/python2.7/IB.py`
   - `swig/IB.2.20211006.4.0a.py`
   - `swig/IB.py`
2. (2 files)
   - `embeddings/include/IO.hpp`
   - `embeddings/include/io.h`
3. (2 files)
   - `embeddings/include/LSMVectorStorage.h`
   - `embeddings/unified/LSMVectorStorage.h`
4. (2 files)
   - `embeddings/include/hf_mapper.hpp`
   - `embeddings/unified/hf_mapper.hpp`
5. (2 files)
   - `embeddings/include/hf_mapper_format.hpp`
   - `embeddings/unified/hf_mapper_format.hpp`
6. (2 files)
   - `embeddings/include/unified_hnsw.hpp`
   - `embeddings/unified/unified_hnsw.hpp`
7. (2 files)
   - `embeddings/test/run_test.sh`
   - `embeddings/tests/run_test.sh`
8. (2 files)
   - `embeddings/test/test-pipe`
   - `embeddings/tests/test-pipe`
9. (2 files)
   - `embeddings/tests/LSMVectorStorage.cpp`
   - `embeddings/unified/LSMVectorStorage.cpp`
10. (2 files)
   - `embeddings/tests/cosine_test.cpp`
   - `embeddings/unified/cosine_test.cpp`
11. (2 files)
   - `embeddings/unified/train_binary.cpp`
   - `embeddings/utils/train_binary.cpp`
12. (2 files)
   - `filters/xpdf-3.01-bsn/xpdf/TextOutputDev.cc`
   - `filters/xpdf-3.01-bsn/xpdf/XMLOutputDev.cc`
13. (2 files)
   - `lib/python2.7/IB.2.20210703.4.0a.py`
   - `swig/old/IB.2.20210703.4.0a.py`
14. (2 files)
   - `swig/old/IB.2.20081127.3.7a.py`
   - `swig/old/IB.2.20090318.3.8a.py`

## Code smell indicators (top files)

### FIXME/TODO/HACK
- Total matches: 346
- `other-code/magic/file-4.26/configure`: 65
- `doctype/mailfolder.cxx`: 22
- `doctype/sgmlnorm.cxx`: 19
- `embeddings/include/ggml.h`: 15
- `other-code/magic/file-4.26/magic/Magdir/archive`: 10
- `src/index.cxx`: 10
- `other-code/magic/file-4.26/src/getopt_long.c`: 9
- `src/httplib.h`: 9
- `filters/xpdf-3.01-bsn/configure`: 5
- `other-code/date/date.cxx`: 5
- `src/date.cxx`: 5
- `docs/re-Isearch-Handbook.md`: 4
- `doctype/antihtml.cxx`: 4
- `doctype/doctype.cxx`: 4
- `other-code/magic/file-4.26/libtool`: 4
- `other-code/magic/file-4.26/ltmain.sh`: 4
- `other-code/magic/file-4.26/src/apprentice.c`: 4
- `src/defs.cxx`: 4
- `src/infix2rpn.cxx`: 4
- `doctype/pdfdoc.cxx`: 3

### preprocessor #if 0 blocks
- Total matches: 320
- `src/index.cxx`: 29
- `src/lang-codes.cxx`: 13
- `src/idb.cxx`: 10
- `filters/xpdf-3.01-bsn/xpdf/TextOutputDev.cc`: 8
- `filters/xpdf-3.01-bsn/xpdf/XMLOutputDev.cc`: 8
- `src/irset.cxx`: 8
- `src/mdt.cxx`: 8
- `doctype/gilsxml.cxx`: 7
- `src/string.cxx`: 6
- `embeddings/src/BertIndex.cpp`: 5
- `src/isearch_main.cxx`: 5
- `src/utf8-B.cxx`: 5
- `src/utf8.cxx`: 5
- `swig/ib-2009.i`: 5
- `swig/ib-2020.i`: 5
- `swig/ib.i`: 5
- `doctype/autodetect.cxx`: 4
- `doctype/doctype.cxx`: 4
- `doctype/htmlmeta.cxx`: 4
- `embeddings/main.cpp`: 4

### catch (...) usage
- Total matches: 76
- `src/index.cxx`: 10
- `src/irset.cxx`: 7
- `src/mdt.cxx`: 6
- `embeddings/unified/LSMVectorStorage-NEW-2.h`: 4
- `embeddings/unified/LSMVectorStorage-NEW.h`: 4
- `src/idb.cxx`: 4
- `src/strlist.cxx`: 4
- `src/thesaurus.cxx`: 4
- `embeddings/utils/config_editor.cpp`: 3
- `src/string.cxx`: 3
- `embeddings/include/LSMVectorStorage.h`: 2
- `embeddings/unified/LSMVectorStorage.h`: 2
- `src/buffer.cxx`: 2
- `src/fpt.cxx`: 2
- `src/gpolyfield.cxx`: 2
- `src/httplib.h`: 2
- `embeddings/main.cpp`: 1
- `embeddings/old/main.cpp`: 1
- `embeddings/src/BertIndex.cpp`: 1
- `src/common.cxx`: 1

### raw new usage
- Total matches: 2212
- `swig/ib_pythonwrap.cxx`: 201
- `swig/java/ib_javawrap.cxx`: 189
- `swig/ib_python64wrap.cxx`: 134
- `src/dtreg.cxx`: 90
- `filters/xpdf-3.01-bsn/xpdf/GfxState.cc`: 54
- `filters/xpdf-3.01-bsn/xpdf/GlobalParams.cc`: 50
- `src/index.cxx`: 47
- `filters/xpdf-3.01-bsn/xpdf/SplashOutputDev.cc`: 43
- `filters/xpdf-3.01-bsn/goo/gfile.cc`: 41
- `filters/xpdf-3.01-bsn/xpdf/JBIG2Stream.cc`: 41
- `filters/xpdf-3.01-bsn/xpdf/PSOutputDev.cc`: 33
- `filters/xpdf-3.01-bsn/xpdf/TextOutputDev.cc`: 28
- `filters/xpdf-3.01-bsn/xpdf/XMLOutputDev.cc`: 28
- `filters/xpdf-3.01-bsn/xpdf/Gfx.cc`: 26
- `filters/xpdf-3.01-bsn/xpdf/PDFCore.cc`: 24
- `docs/re-Isearch-Handbook.md`: 20
- `src/irset.cxx`: 20
- `src/vidb.cxx`: 19
- `filters/xpdf-3.01-bsn/xpdf/Stream.cc`: 17
- `src/string.cxx`: 17

### goto usage
- Total matches: 1271
- `swig/ib_python64wrap.cxx`: 665
- `filters/xpdf-3.01-bsn/xpdf/GfxState.cc`: 51
- `filters/xpdf-3.01-bsn/xpdf/Function.cc`: 40
- `filters/xpdf-3.01-bsn/xpdf/JBIG2Stream.cc`: 36
- `filters/xpdf-3.01-bsn/xpdf/XRef.cc`: 34
- `other-code/magic/file-4.26/src/ascmagic.c`: 32
- `filters/xpdf-3.01-bsn/xpdf/Stream.cc`: 27
- `src/malloc-2.8.3.c`: 27
- `src/malloc.c`: 27
- `filters/xpdf-3.01-bsn/xpdf/Link.cc`: 25
- `other-code/sru-srw/cql/cql.c`: 24
- `src/unicodedata.c`: 21
- `filters/xpdf-3.01-bsn/xpdf/Gfx.cc`: 20
- `other-code/magic/file-4.26/src/apprentice.c`: 20
- `filters/xpdf-3.01-bsn/xpdf/JPXStream.cc`: 18
- `src/setlocale.c`: 17
- `filters/xpdf-3.01-bsn/xpdf/GlobalParams.cc`: 11
- `other-code/magic/file-4.26/src/magic.c`: 11
- `filters/xpdf-3.01-bsn/xpdf/SplashOutputDev.cc`: 10
- `filters/xpdf-3.01-bsn/xpdf/GfxFont.cc`: 8

## Notes
- These indicators are triage aids and not automatic defects.
- Third-party/vendor trees account for a significant fraction of smell counts.
