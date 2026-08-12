# Porting del backend ROCm di DwarfStar (ds4) da wave32/gfx1151 a wave64/gfx906

> Nota: questo documento conserva il diario storico dell'indagine iniziale e
> non e' il piano di rilascio. Le affermazioni successive sulla serializzazione,
> sulla mappa GPU e sui target di contesto sono state superate dai test finali.
> Per lo stato verificato usare `PULL_REQUEST-GFX906.md`; per hardware, SSD e
> roadmap usare `SPEEDUP-GFX906.md`.

Piano di porting per far girare ds4 (backend ROCm) su GPU AMD gfx906
(Vega 20: Radeon VII, Radeon Pro VII, MI50/MI60) che hanno **wavefront 64
fisso**, laddove il backend è scritto per **wave32** (gfx1151 / Strix Halo).

Basato su analisi del codice al `main` (commit attuale).

---

## 1. Contesto e analisi del codice

### 1.1 Perché oggi non compila
- `ds4_rocm.h` include `<rocwmma/rocwmma.hpp>`.
- rocWMMA 7.2 (`/opt/rocm/include/rocwmma/internal/config.hpp:54-61`) attiva il
  path solo per `gfx908/90a/942/950/1100/1150/1151/1200/1201`; **gfx906 manca** →
  `static_assert(0, "Unsupported architecture")` in `config.hpp:79`.
- gfx906 non ha i Matrix Core MFMA (introdotti da gfx908/MI100). Non sono
  "tensor file" mancanti come per rocBLAS (che invece è già patchato su questo
  sistema: 156 file `*gfx906*` in `/opt/rocm/lib/rocblas/library/`).

### 1.2 Perché non basta "aggiustare rocWMMA"
Il backend è **pervasivamente wave32**, non solo rocWMMA. Misurato sul codice:

| Punto | Rilevamento | Dove |
|---|---|---|
| `tid >> 5` (wave = 32 thread) | 94+ siti | q8: 58, moe: 19, attention: 7, common: 6, indexer: 3, norm_rope: 1 |
| Kernel `_w32` / `_row32` | 27 | q8: 13, moe: 12, common: 2 |
| Varianti `_w64` | **0** | nessuna |
| `__launch_bounds__` espliciti | 2 | q8:682 `(128,2)`, moe:3095 `(32)` |
| blockDim launch tipico | `<<<grid, 256>>>` | 256 = 8 wave32 o 4 wave64 |
| Primitive wave-level (shfl/DPP/ballot) | 32 siti | attention: 11, q8: 8, moe: 6, common: 4, router: 3 |
| rocWMMA `fragment<...>` | 8 siti | moe: 6, q8: 2 |
| Esplicito "gfx1151" | 1 commento | `ds4_rocm_matmul.cuh:672` |

gfx906 (GCN5) ha **wavefront 64 fisso**, non supporta wave32 hardware.

---

## 2. Strategia: emulazione wave32 su hardware wave64

Invece di riscrivere i 27 kernel a wave64 reale (94+ siti, ricalcolo di
MTILES/grid/blockDim, alto rischio regression), **si mantiene la convenzione
wave32 software** (`wave = tid>>5`, `lane = tid&31`) e la si emula su wave64:

- **256 thread/block = 4 wavefront hardware da 64**; ognuno contiene **2
  "wave32 software"** (tid32 0..31 e 32..63). I due gruppi fanno lavoro
  indipendente (tile diversi) nello stesso wavefront → divergenza controllata,
  FMA scalar regolari, shared mem per-thread → OK.
- **`__shfl`/DPP/ballot** (32 siti): su wave64 operano su 64 lane. Per restringerli
  a 32 si usa `__shfl(val, srcLane, /*width=*/32)` (HIP supporta `width`
  potenza di 2) e maschere lane `0x1F`. Emulazione wave32 **nativa su ROCm**.
- **rocWMMA** (8 siti, subset minimo): si sostituisce con uno **shim device-side**
  che implementa la stessa API (`fragment`, `load/fill/mma/store_matrix_sync`)
  per `half×half→float` con tile 16×16, usando 32 thread/wave32-software con
  cooperative shared-mem GEMM (256 elementi / 32 thread = 8 elementi/thread).
  Bypassando `<rocwmma/rocwmma.hpp>` via `#if defined(__gfx906__)` in
  `ds4_rocm.h`, il `static_assert` non scatta.

**Vantaggi**: il 90% del codice kernel non si tocca. Rischi concentrati in 2
aree circoscritte (shim rocWMMA + 32 siti wave-level).

**Svantaggi**: si spreca metà wavefront (32 lane inattive per tile). Performance
stimata ~30-50% del rocWMMA-su-MI100. Se insufficiente, M6 valuta porting
wave64 reale sui soli kernel hot.

---

## 3. Milestone (M0 → M6)

### M0 — Probing build ✅ (fatto)
`make rocm ROCM_ARCH=gfx906` fallisce in `rocwmma/internal/config.hpp:79`.
Confermato: rocWMMA è l'unico blocco compile-time; il resto è runtime.

### M1 — Shim rocWMMA device-side per gfx906 (sblocca la build)
- **Crea** `rocm/ds4_rocm_wmma_gfx906.cuh`:
  - `namespace rocwmma` shim con le **sole API usate**: `fragment<matrix_a|matrix_b|accumulator,16,16,16, half|float, row_major|col_major>`,
    `load_matrix_sync`, `fill_fragment`, `mma_sync`, `store_matrix_sync`,
    enum `matrix_a/matrix_b/accumulator`, `row_major/col_major/mem_row_major`.
  - `fragment` = struct con storage interno row_major (16×16 half per A/B,
    16×16 float per accumulator) allocato in **shared mem per-wave32-software**.
  - `load_matrix_sync`: copia il tile da `ptr` (con `ldm`, layout) → storage
    interno. Gestisce `row_major`/`col_major` in fase di load (trasporto).
  - `mma_sync(d,a,b,c)`: `d = a·b + c`, cooperative GEMM 16×16 su 32 thread
    (lane = `tid&31`), 8 elementi C/thread, FMA scalar half→float.
  - `fill_fragment(frag, v)`, `store_matrix_sync(ptr, frag, ldm, layout)`.
  - `#if defined(__gfx906__)` per attivarlo solo su Vega 20.
- **Patch** `ds4_rocm.h`: prima di `#include <rocwmma/rocwmma.hpp>`,
  `#if defined(__gfx906__) #include "rocm/ds4_rocm_wmma_gfx906.cuh" #else ... #endif`.
  (Attenzione: anche `ds4_rocm.cu` include rocwmma via `ds4_rocm.h`.)
- **Verifica**: `make rocm ROCM_ARCH=gfx906 -j$(nproc)` deve superare la fase
  `ds4_rocm.o` (rocWMMA bypassato). Possono emergere errori wave-level → M2.
- **Deliverable**: `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`
  compilano per gfx906.

### M2 — Primitive wave-level a width=32 (semantica wave32 su wave64)
- **Identifica** i 32 siti (elenco in appendice A.3):
  - `__shfl*` → `__shfl*(val, src, /*width=*/32)` o maschera `0xFFFFFFFF` sui
    primi 32 lane.
  - DPP (`dpp_row_*`) → `row_shl/row_shr` con lane mask `0x1F` (lane < 32).
  - `__ballot`/`__any`/`__all` → maschera a 32 lane.
- **Patch in-place** nei 5 file (attention 11, q8 8, moe 6, common 4, router 3).
- **Verifica**: build pulita; `make` senza warning wave-level.
- **Deliverable**: nessun sito wave-level presume wave32 implicito.

### M3 — Correttezza numerica
- `make test` → `ds4-eval --self-test-extractors` deve passare.
- Confronto logits prompt breve vs build di riferimento (CPU o CUDA su altra
  macchina). Tolleranza: match entro 1e-3 relativo sui top-k.
- Se drift > tol: debug kernel per kernel (shfl width, shim mma accumulo).
- **Deliverable**: logits match entro tolleranza.

### M4 — Smoke test inference su 1 GPU
- Libera 1 GPU 16 GB (es. Pro VII device 0). `rocm-smi --showmeminfo vram`.
- Lancia `./ds4 -m <IQ2XXS.gguf> --rocm --ctx 4096 -p "Say DS4_READY."`
  con `ROCR_VISIBLE_DEVICES=0`.
- Verifica output coerente e no crash.
- **Deliverable**: inferenza corretta su singola gfx906 16 GB.

### M5 — Distributed PP6 (6 GPU) + `run.sh`
- Adotta `run.sh` già scritto (coordinator su MI50 device 3, 5 worker).
- Verifica pipeline distribuita: `rocm-smi -l 1` mostra tutte e 6 le GPU attive
  durante il prefill (non dente di sega).
- Benchmark `ds4-bench` 4096 token, 96K ctx. Target: > prefill in-process.
- **Deliverable**: 6 GPU in pipeline, prefill accelerato, decode corretto.

### M6 (opzionale) — Performance / porting wave64 reale
- Se M5 rivella collo di bottiglia nell'emulazione wave32 (metà wavefront
  sprecato), valuta porting wave64 reale dei **soli kernel hot** (MoE Q2K + Q8
  matmul principale): `tid>>5`→`tid>>6`, `&31`→`&63`, MTILES 8→4, ricalcolo grid.
- Deliberable: benchmark vs M5; merge solo se guadagno > 30%.

---

## 4. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Shim rocWMMA numericamente divergente | M3 confronto logits; accumula in float, double-check layout half↔float |
| DPP non riducibile a 32 lane su gfx906 | gfx906 DPP supporta `row_shl/row_shr` con lane mask → ok; fallback shfl |
| LDS shared mem per-wave32-software cresce | 4 wave64 × 2 wave32 × (A+B+C tile) = ~6 KB; Vega20 64 KB LDS/CU → ok |
| Occupancy bassa (wave64, 4 wave/block) | `__launch_bounds__` da ritoccare; M6 se serve |
| Engine ↔ checkpoint accoppiati | invariato rispetto a recipe 3090 |

---

## 5. Appendice A — Dati di dettaglio

### A.1 Kernel `_w32` / `_row32` da adattare (27)
```
ds4_rocm_q8.cuh:13    matmul_q8_0_*_w32 / shared_gate_up_swiglu_q8_0_*_w32 / grouped_q8_0_*_w32
ds4_rocm_moe.cuh:12   moe_gate_up_mid_*_row32 / moe_down_*_row32 / moe_*_q2K_rows_w32
ds4_rocm_common.cuh:2 matmul_f16_*_w32
```
Con emulazione wave32 (strategia §2) **non vanno riscritti**: va verificato solo
che non usino shfl/DPP senza width (→ M2).

### A.2 rocWMMA — 8 siti (subset API minimo)
```
ds4_rocm_moe.cuh:3580, 3726, 3868, 3958   (4 frammenti MoE, half/half/float, 16x16)
ds4_rocm_q8.cuh:802, 1595                   (2 frammenti Q8, half/half/float, 16x16)
```
Tutte `fragment<matrix_a|b|accumulator, 16, 16, 16, half|float, row_major|col_major>`.

### A.3 Primitive wave-level (32 siti)
```
ds4_rocm_attention.cuh:11   ds4_rocm_q8.cuh:8   ds4_rocm_moe.cuh:6
ds4_rocm_common.cuh:4        ds4_rocm_router.cuh:3
```
Pattern: `__shfl*`, DPP (`dpp_row_*`), `__ballot`. Su wave64 vanno ristretti a
width=32 / lane mask 0x1F.

---

## 6. Stato
- [x] M0 probing build
- [x] M0.1 validazione strategia (shfl già `width=32` nel codice → M2 quasi gratis)
- [x] M1 shim rocWMMA + build (`rocm/ds4_rocm_wmma_gfx906.cuh`, bypass in
      `ds4_rocm.h` + `ds4_rocm_q8.cuh` + `ds4_rocm_moe.cuh`, Makefile
      `-DDS4_GFX906` quando `ROCM_ARCH=gfx906`). **Compila e linka** i 5 binari.
- [x] M1.1 esclusione `matmul_q8_0_f32_batch_wmma_4w_kernel` (usa
      `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, hardcoded gfx11/wave32):
      su gfx906 il dispatch cade su `cuda_launch_q8_batch_sharedx`.
- [x] M2 primitive wave-level (in parte già `width=32`; vedi M2.1/M2.2 per i bug reali)
- [x] M3 correttezza numerica — self-test extractors passa (NB: testa solo il
      parsing, non i kernel GPU; la validazione kernel vera è M4/M5)
- [x] M4 smoke test inference — **funziona**: output coerente e corretto
      ("1 2 3 ... 30", reasoning DS4_READY) su pipeline distribuita a 6 GPU
- [x] M5 distributed PP6 + run.sh — pipeline attiva, API OpenAI su :8000,
      prefill ~40 t/s, decode ~6.8 t/s (con workaround serializzazione, vedi §7)

## 7. Bug wave32→wave64 trovati e risolti (runtime)

Oltre al blocco compile-time di rocWMMA (M1), il porting ha richiesto 3 fix runtime,
tutti nella classe "sync mask / broadcast calcolate per wavefront 32":

1. **Sync mask di sottogruppo (CRASH)** — `half_warp_sum_f32` / `quarter_warp_sum_f32`
   (`ds4_rocm_moe.cuh:470,479`) e 2 siti in `ds4_rocm_attention.cuh:720,965`
   calcolavano la mask come `0xff << (threadIdx.x & 24)` (offset dentro un
   wavefront da 32). Su wave64 le lane 32-63 sono nello STESSO wavefront hardware
   delle lane 0-31 → mask invalida → `HSA_STATUS_ERROR_EXCEPTION` (0x1016).
   Fix: mask allineata alla lane assoluta nel wavefront reale via
   `__builtin_amdgcn_wavefrontsize()` e `MASK_T` a 64 bit (`0xff << (wlane & ~7)`).

2. **Router sync mask (CRASH potenziale)** — `ds4_rocm_router.cuh:90-92` usava
   `__shfl_xor_sync(FULL_WARP_MASK, ...)` con `block(32,4)`: su wave64 due righe
   da 32 lane condividono un wavefront; su decode le righe inattive escono prima
   → mask a 64 bit richiede lane inattive → trap. Fix: mask per-riga
   (`0xffffffff << (lane_abs & ~31)`).

3. **Broadcast senza width (NUMERICO)** — `__shfl_sync(FULL_WARP_MASK, x, 0)` in
   `ds4_rocm_attention.cuh:1159,1270,1282,1435,1471` (width di default =
   warpSize=64 su gfx9) trasmetteva dalla lane 0 assoluta a tutte le 64 lane →
   la metà superiore di ogni wavefront riceveva lo score sbagliato → logits
   degenerate (BOS ripetuto). Fix: `width=32` (broadcast per-segmento; no-op su
   wave32/CUDA).

Diagnosi effettuata con `AMD_SERIALIZE_KERNEL=3` (attribuzione errore al kernel)
e con `tests/shim_test.cu` (validazione standalone dello shim: `max_err=0`).

## 8. Known issue: race condition residua (workaround attivo)

**Sintomo**: senza serializzazione, l'output decade (logits costanti → token BOS
ripetuto); con serializzazione l'output è corretto. Indica una race
kernel/kernel o kernel/copy nel backend ROCm, mascherata su gfx1151 dal timing
ed esposta su gfx906.

**Bisezione (2026-07-17)**: serializzando **solo il coordinator** (`ds4-server`,
che fa embedding + layer 0:12 + output head + sampling + orchestrazione TCP) e
lasciando i 5 worker **completamente liberi**, l'output è corretto. → la race è
**sul coordinator**, nel path compute/output, non sui worker né nel trasporto
TCP (i readback/upload tensor sono `cudaMemcpy` sincroni e `ds4_gpu_synchronize`
fa `cudaDeviceSynchronize` prima degli invii; il path SSD-streaming non è attivo
in modalità residente).

**Workaround attivo in run.sh**: `AMD_SERIALIZE_KERNEL=1` **solo sul coordinator**
(i worker non la impostano). Costo contenuto, output corretto.

**Impatto prestazioni**:
- serializzazione completa (tutti i processi): decode ~6.8 t/s, prefill ~40 t/s
- **solo coordinator** (config adottata): **decode ~10.5 t/s, prefill ~57 t/s**
  (+46% decode, +43% prefill)
- la rimozione completa richiede il fix della race a monte (da fare upstream con
  strumentazione dedicata, es. event tracing sul path output head del coordinator)


## 9. Mappatura device (critica per run.sh)

**ROCR device index = rocminfo node (N+1), NON la colonna "GPU[N]" di rocm-smi.**
Verificato via lspci + report dei nomi dai processi:

| ROCR dev | node | GPU | VRAM | ruolo in run.sh |
|---|---|---|---|---|
| 0 | 1 | Radeon VII | 16 GB | worker 13:18 |
| 1 | 2 | Pro VII | 16 GB | worker 19:24 |
| 2 | 3 | **MI50** | **32 GB** | **coordinator 0:12 (+output)** |
| 3 | 4 | Pro VII | 16 GB | worker 25:30 |
| 4 | 5 | Pro VII | 16 GB | worker 31:36 |
| 5 | 6 | Radeon VII | 16 GB | worker 37:42 |

Layer split ribilanciato: coordinator 13 layer (~25.4 GiB su 32 GB), 5 worker
6 layer (~11 GiB, ~3.5 GiB liberi su 16 GB). ctx 32768. Arena 256 MiB
(`DS4_ROCM_WEIGHT_ARENA_CHUNK_MB`, dalla patch dedicata).

## 10. Risultati misurati (2026-07-17, PP6 gfx906, ctx 32K)

| Metrica | Valore |
|---|---|
| Build | ✅ 5 binari per gfx906 |
| API OpenAI | ✅ `http://127.0.0.1:8000` |
| Qualità output | ✅ coerente/corretto (workaround §8) |
| Prefill (811 tok) | ~57 t/s |
| Decode | **~10.5 t/s** (serializzazione solo coordinator; ~6.8 t/s con serializzazione completa) |
| GPU utilizzate | 6/6 (5×16GB worker + MI50 32GB coordinator) |

Confronto recipe 4× RTX 3090 (170/23 t/s): gfx906 è più lento per età architettura
(no MFMA, emulazione FMA-scalar via shim), emulazione wave32 (metà wavefront
inutilizzata) e workaround serializzazione. Miglioramenti futuri in §M6 e §8.

- [ ] M3 correttezza numerica
- [ ] M4 smoke test 1 GPU
- [ ] M5 distributed PP6 + run.sh
- [ ] M6 (opz.) porting wave64 hot kernels

Patch già pronta per le versioni future:
`ds4-rocm-weight-arena-chunk-env.patch` (parità ROCm/CUDA weight-arena env var,
indipendente dal porting, si applica a backend ROCm funzionante).
