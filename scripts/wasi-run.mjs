// Run a wasm32-wasip1 binary under Node's WASI with the current directory mounted.
// Usage: node scripts/wasi-run.mjs <file.wasm> [args...]
import { readFile } from 'node:fs/promises'
import { WASI } from 'node:wasi'

const [wasmPath, ...args] = process.argv.slice(2)
const wasi = new WASI({
  version: 'preview1',
  args: [wasmPath, ...args],
  preopens: { '.': '.' },
})
const module = await WebAssembly.compile(await readFile(wasmPath))
const instance = await WebAssembly.instantiate(module, wasi.getImportObject())
process.exitCode = wasi.start(instance)
