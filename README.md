# x86-recomp

A static recompiler for 32-bit Windows programs (PE32, i386, MSVC-built): every function of
the executable becomes a C function, so the program runs natively on x86-64 or aarch64 Linux
on top of native replacements for the Windows APIs it imports.

It is the shared layer under per-program projects, the way gcn-recomp is for GameCube games:

- kintsugi-symphonia: *Tales of Symphonia* (Steam, 2016), where it was first written

**This repository contains no program code or data.** A project builds from its own copy of the
executable, and the generated C lands in that project's build directory. `tools/check_generic.py`
fails on any program's name or distinctive nouns (`tools/game_terms.txt`) outside this file.

## Pieces

- `recomp/pe.py`: loads the PE and the project's config (`sha256`, optional `entry`,
  `code_sections`, an `iat` override for programs behind a DRM wrapper, extra `noreturn`
  imports). The format is described at the top of the file.
- `recomp/discover.py`: finds the functions. Recursive descent from the entry; seeds from call
  targets, code addresses used as immediates and stored in data sections (vtables, callbacks),
  and function-shaped starts in leftover gaps. MSVC jump tables (`jmp [r*4+T]`, also with a
  byte index table) are sized from their bounds check; no-return functions are found by
  fixpoint. Output: `functions.json`, `gaps.json` and a coverage line.
- `recomp/x86.py` + `recomp/recomp.py`: the translator. Registers are C locals inside a
  function and synced with `struct cpu` around calls. Flags are lazy (operation + operands,
  folded by the C compiler where the consumer is known). x87 is a runtime stack in doubles,
  with precision control (24-bit rounds to float) and rounding control (fist/frndint).
  MMX/SSE go through helpers in `runtime/simd.h`. Jump tables become `switch`, indirect calls
  and jumps go through `dispatch()`, `call [IAT slot]` calls `imp_<dll>_<name>(c)`.
  Undecodable or unsupported instructions become `trap()` calls with a message.
  Output: `funcs_NNN.c`, `funcs.h`, `table.c` (address → function), `imports.h`/`imports.txt`,
  `simd_used.txt`, `report.txt`.
- `recomp/fuzz.py`: differential test against Unicorn. It takes real instances of every
  instruction form the program uses, runs them from random states through Unicorn and through
  the translator's C, and compares registers, defined flags, memory writes, x87 and vector state.
- `recomp/census.py`: mnemonic counts over the discovered code.
- `runtime/`: `cpu.h` (CPU state, memory access, lazy-flag helpers), `cpu.c` (x87 control word,
  80-bit format, `rep movs/stos`), `simd.h`, `table.h`.

Functions listed under `"replace"` in the config get no generated body: the project defines
`f_XXXXXXXX(struct cpu *)` itself. That is how recompiled code is rewritten as native code one
function at a time, with the rest of the program unchanged.

Instructions listed under `"hooks"` are preceded by a call to the project's
`int (*insn_hook)(struct cpu *, u32 addr)` with the registers synced to `struct cpu`: the hook
can read or change any state (registers, x87 stack, memory), and a nonzero return skips the
instruction (a skipped `call` pushes and calls nothing). That is for changing a program's
behaviour in a few places without rewriting whole functions.

The project supplies the rest of the runtime: loader, `dispatch()`, `trap()`, threads/TEB and
the import shims.

## Memory model

Guest addresses are host addresses: the runtime maps the image at its preferred base and keeps
every guest allocation in the low 4 GB, so a guest pointer is `(void *)(uintptr_t)addr`. That
needs an executable without relocations at a fixed base (true for typical MSVC .exe files) and a
host that leaves the low 4 GB free (Linux PIE binaries on x86-64 and aarch64 do). All access
goes through `ld*/st*` in `cpu.h`, so a base offset could be introduced in one place.

## Usage

```
python3 -m venv .venv && .venv/bin/pip install capstone pefile unicorn
.venv/bin/python -I recomp/discover.py CONFIG.json PROGRAM.exe OUT
.venv/bin/python -I recomp/recomp.py   CONFIG.json PROGRAM.exe OUT/functions.json GEN
.venv/bin/python -I recomp/fuzz.py     CONFIG.json PROGRAM.exe OUT/functions.json WORK [--per-form 6 --states 6]
```

## Known limits (what the next program will probably need)

- SIMD: only the MMX/SSE forms seen so far have helpers (the build fails on a missing one).
  Programs built with `/arch:SSE2` will need many more, and scalar SSE math through them.
- C++ exceptions / SEH: `fs:` addressing works, but unwinding through `RaiseException` /
  `_CxxThrowException` is not implemented. Programs that throw in normal operation need it.
- No self-modifying or runtime-generated code. DLLs the program ships with need their own
  translation, or native replacements.
- x87 runs in doubles. Code that relies on 80-bit range or precision will differ.
- Not yet fuzzed: branches as such, overlapping `rep movs`, x87 in 24-bit precision mode.

## License

MIT (see `LICENSE`). This covers the code in this repository only, not any game or program it is used with.
