# Piano operativo di ottimizzazione ds4 su gfx906

## Risultato dell'analisi

La configurazione piu' efficace e affidabile su questa macchina e' una pipeline
PP6 completamente residente. La GPU da 32 GB esegue coordinator, layer 0:12 e
output head; le cinque GPU da 16 GB eseguono sei layer ciascuna. La GPU usata
dalla GUI non viene liberata: resta nella pipeline con soli sei layer e con un
margine VRAM maggiore rispetto a una divisione a sette layer.

Hardware rilevato il 18 luglio 2026:

| Risorsa | Configurazione | Implicazione |
|---|---|---|
| CPU | AMD Threadripper PRO 3955WX, 16 core / 32 thread, 1 nodo NUMA | sufficiente per orchestrazione e I/O; non e' il collo di bottiglia decode |
| RAM | circa 96 GB | puo' contenere il GGUF da circa 81 GiB, ma con poco margine per cache duplicate |
| GPU | 6 gfx906, 60 CU wave64: 5 x 16 GB + 1 x 32 GB | modello distribuito obbligatorio; gfx906 non dispone di MFMA |
| GPU GUI | ROCR 0, 16 GB, Radeon VII | non liberabile; assegnare solo layer 13:18 e sorvegliare temperatura/throttling |
| Coordinator | ROCR 2, 32 GB | layer 0:12, embedding e output head |
| Interconnect | PCIe x16, nessun XGMI | minimizzare hop e trasferire attivazioni fp16 |
| SSD | Samsung 990 Pro, lettura sequenziale osservata circa 5.8 GB/s | utile come capacity tier, non come sostituto della VRAM nel decode corrente |

Mappatura ROCR verificata da `run.sh`:

| ROCR | VRAM | Ruolo residente |
|---:|---:|---|
| 0 | 16 GB, GPU GUI | worker 13:18 |
| 1 | 16 GB | worker 19:24 |
| 2 | 32 GB | coordinator 0:12 + output |
| 3 | 16 GB | worker 25:30 |
| 4 | 16 GB | worker 31:36 |
| 5 | 16 GB | worker 37:42 |

Il profilo consigliato usa `CTX=28672`, `PREFILL_CHUNK=64`,
`DIST_WINDOW=5`, attivazioni distribuite fp16 e arena da 256 MiB. Il ceiling a
28K mantiene margine sulla scheda GUI; 32K va qualificato con un soak termico,
non assunto sicuro.

## Misure e stato di correttezza

- Build gfx906 completa e suite GPU isolate passate.
- Shim rocWMMA: `max_err=0` su 8192 valori / 256 tile.
- Regressione long-context: top-k 32768 x 32 e path attention oltre 8192 passati.
- Prompt reale da 21.718 token: prefill circa 70.5 token/s, decode circa 8.36
  token/s, output coerente.
- Prompt breve residente senza `AMD_SERIALIZE_KERNEL`: decode misurato fino a
  12.29 token/s. La serializzazione globale non e' quindi il default.
- La GPU GUI e' il worker piu' sensibile al throttling: sotto carico prolungato
  il tempo per chunk puo' crescere da circa 0.4 s a circa 1.0 s.

Un test con un budget di soli 16/32 token puo' terminare durante il reasoning e
non e' un gate di qualita' valido. Le prove E2E di merge devono concedere almeno
128 token e verificare contenuto deterministico, valori finiti e assenza di
sequenze BOS ripetute.

## SSD streaming: cosa e' utilizzabile e cosa manca

Lo streaming e' stato valutato esplicitamente. La patch rende la riserva VRAM
proporzionale alla scheda (la precedente riserva fissa da 16 GiB impediva ogni
cache su una GPU da 16 GB), evita di preparare due volte gli span statici gia'
installati e aggiunge telemetria per le mappe/cache.

Non e' pero' corretto abilitare oggi un profilo ibrido PP6 in produzione. Nei
worker con slice che iniziano da un layer diverso da zero il fallback mappa il
layer completo e finisce la VRAM. Forzando la tabella selected-address si evita
l'OOM, ma il MoE batch pointer-table ROCm/gfx906 produce attivazioni non finite
e, a valle, expert id `-1`. Sincronizzazioni aggiuntive e letture SSD dirette non
hanno risolto il difetto. `run.sh` mantiene quindi il profilo residente.

Gate necessario prima di offrire SSD distribuito:

1. test isolato di equivalenza del MoE batch tra pesi residenti e tabella di
   puntatori, con batch 2, 8, 32 e 64;
2. controllo `isfinite` dopo gate/up/down e dopo ogni layer;
3. test di una slice con `layer_start != 0` senza mappatura full-layer;
4. soak da almeno 28K con eviction della cache e confronto greedy col residente.

L'SSD resta una leva di capacita': anche a 5.8 GB/s non puo' alimentare tutti
gli expert ad ogni token senza una cache con hit-rate elevato. Il percorso
residente e' quello che usa realmente il 100% delle sei GPU nel modello attuale.

## Roadmap operativa

### P0 - Merge di correttezza e riproducibilita'

- Integrare shim gfx906, maschere wave64-safe, broadcast con width 32, guardia
  del builtin gfx11 e selezione esplicita `ROCM_ARCH`.
- Integrare `make gfx906` e `make rocm-regression ROCM_ARCH=gfx906`.
- Usare `run.sh` come configurazione host-specific, senza presumere che gli
  indici ROCR siano uguali agli indici `rocm-smi`.
- Gate: build pulita, shim `max_err=0`, long-context smoke, PP6 greedy >=128
  token e prompt >=20K.

### P1 - Generalizzazione a tutti i sistemi gfx906

- Rilevare a startup nome, VRAM, CU, GPU occupata dalla GUI e memoria libera.
- Pianificare i layer per capacita' effettiva: embedding/output sulla GPU piu'
  capiente, almeno 4 GiB di riserva sulle 16 GB, penalita' per GPU GUI/calde.
- Generare una mappa suggerita, ma permettere override esplicito.
- Qualificare Radeon VII, Pro VII, MI50 e MI60 con ROCm supportato dal sistema.
- Non usare `DS4_GFX906` per altre architetture: il build generico ROCm conserva
  rocWMMA e i path nativi.

### P2 - Correzione SSD selected-address

- Aggiungere il test di equivalenza descritto sopra prima di modificare il
  launcher.
- Isolare il primo kernel che genera non-finiti confrontando output residenti e
  pointer-table.
- Solo dopo il pass, aggiungere un profilo `DS4_PROFILE=hybrid` e misurare
  cache-hit, byte letti, stall I/O e qualita'.

### P3 - Kernel gfx906 nativi wave64

- Profilare con rocprof i kernel MoE e Q8; ottimizzare prima il maggiore tempo
  cumulativo, non il numero di chiamate.
- Sostituire progressivamente i gruppi wave32 emulati con tile wave64 nei path
  caldi, oppure impaccare 2/4 half per shuffle nello shim.
- Gate per ogni kernel: confronto numerico, regressione long-context, guadagno
  >=10% sul kernel e nessuna regressione E2E.

### P4 - GEMM grandi e pipeline PCIe

- Valutare hipBLAS/rocBLAS per proiezioni dense e output head con batch/prefill
  grandi; mantenere i kernel custom per i piccoli GEMM per-expert.
- Conservare attivazioni inter-stage a 16 bit e prefetch depth 2.
- Provare `DIST_WINDOW=6..8` solo se il margine VRAM osservato resta >=1 GiB;
  window 10 ha gia' causato OOM sulle 16 GB.

### P5 - Qualifica termica e contesto

- Soak di 30 minuti a 28K con `rocm-smi` per temperatura, clock, potenza, VRAM
  e utilizzo per GPU.
- Se ROCR 0 throttla, spostare un layer dalla GPU GUI alla 32 GB soltanto se il
  margine coordinator lo consente; altrimenti ridurre clock/power cap per
  stabilita', non aumentare il window.
- Qualificare 32K solo dopo pass senza OOM, non-finiti o perdita di throughput
  superiore al 15% tra inizio e fine soak.

## Comandi di accettazione

```bash
make gfx906 -j$(nproc)
ROCR_VISIBLE_DEVICES=2 make rocm-regression ROCM_ARCH=gfx906

# profilo host verificato
CTX=28672 DIST_WINDOW=5 DS4_COORD_SERIALIZE=0 ./run.sh
```

Una modifica e' accettabile solo se migliora una misura ripetibile e conserva
la correttezza. L'obiettivo “100% hardware” significa mantenere tutte le sei GPU
occupate durante il prefill e ridurre il tempo del tratto sequenziale per token;
non e' realistico attendersi 100% simultaneo nel decode autoregressivo di una
pipeline senza micro-batching di richieste concorrenti.
