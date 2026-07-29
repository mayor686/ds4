# Profilo delle fasi di inferenza su 6x AMD gfx906

## Configurazione misurata

Le statistiche di questo documento provengono da un test eseguito direttamente
sulla configurazione GPU usata dai launcher gfx906:

- modello: DeepSeek-V4-Flash IQ2XXS;
- backend: ROCm 7.2.4, architettura `gfx906` wave64;
- topologia: pipeline parallela PP6;
- coordinatore: MI50 32 GB, layer `0:12` e output head;
- worker: cinque GPU gfx906 da 16 GB;
- suddivisione worker: `13:18`, `19:24`, `25:30`, `31:36`, `37:42`;
- esperti completamente residenti in VRAM, senza streaming SSD;
- contesto configurato: 300.000 token;
- posizione del contesto durante la misura: circa 2.400 token;
- chunk di prefill: 64 token;
- finestra distribuita: 5 chunk;
- trasporto delle attivazioni: FP16;
- prompt: 2.418 token;
- generazione: 64 token.

Il profiler e la telemetria erano abilitati soltanto durante questa misura. I
launcher normali continuano a tenere disattivato il logging della telemetria.

## Ripartizione complessiva della richiesta

| Macro-fase | Tempo | Percentuale sul totale |
|---|---:|---:|
| Prefill | 33,22 s | 86,7% |
| Generazione di 64 token | 5,11 s | 13,3% |
| Tokenizzazione e gestione API | circa 2 ms | <0,1% |
| **Totale richiesta** | **38,33 s** | **100,0%** |

La percentuale complessiva tra prefill e generazione dipende dal rapporto tra
token di input e token generati. Le tabelle successive descrivono separatamente
le due fasi e sono quindi più utili per individuare i colli di bottiglia.

## Prefill

Il prefill usa una pipeline: mentre una GPU elabora un chunk, le altre possono
elaborare chunk precedenti. Di conseguenza, i tempi dei singoli stadi si
sovrappongono e non possono essere sommati per ottenere direttamente il tempo
wall-clock.

La tabella seguente normalizza le percentuali rispetto al lavoro seriale
equivalente di un chunk completo da 64 token, circa 2.829 ms. Grazie alla
sovrapposizione della pipeline, a regime viene completato un chunk ogni circa
819 ms.

| Stadio | Tempo medio per chunk | Percentuale del lavoro seriale | Durata rispetto al bottleneck |
|---|---:|---:|---:|
| Coordinatore, layer `0:12` e dispatch | circa 819 ms | 29,0% | 100,0% |
| Worker ROCR 0, layer `13:18` | 405 ms | 14,3% | 49,4% |
| Worker ROCR 1, layer `19:24` | 392 ms | 13,9% | 47,9% |
| Worker ROCR 3, layer `25:30` | 392 ms | 13,9% | 47,9% |
| Worker ROCR 4, layer `31:36` | 405 ms | 14,3% | 49,4% |
| Worker ROCR 5, layer `37:42` | 390 ms | 13,8% | 47,6% |
| Trasferimenti delle attivazioni tra worker | 25,7 ms | 0,9% | 3,1% |
| **Totale seriale equivalente** | **2.829 ms** | **100,0%** | — |

### Tempo wall-clock del prefill

| Componente | Tempo | Percentuale del prefill |
|---|---:|---:|
| Pipeline a regime | 30,70 s | 92,4% |
| Riempimento iniziale della pipeline | 2,52 s | 7,6% |
| **Totale prefill** | **33,22 s** | **100,0%** |

### Interpretazione del prefill

Il coordinatore è il collo di bottiglia principale. Esegue 13 layer, mentre
ogni worker ne esegue 6, e richiede circa 819 ms per stadio contro 390-405 ms
dei worker. I worker impiegano quindi soltanto il 48-49% circa del tempo dello
stadio più lento.

I trasferimenti tra worker rappresentano circa lo 0,9% del lavoro seriale. La
rete locale non è il limite dominante del prefill; le opportunità principali
sono il riequilibrio dei layer e l'ottimizzazione del calcolo sul coordinatore.

## Generazione autoregressiva di token

Durante il decode i blocchi devono essere attraversati in sequenza per ogni
token. Le percentuali di questa tabella sono quindi direttamente additive e
sono normalizzate rispetto al tempo end-to-end medio di 79,86 ms per token.

| Fase | Tempo medio per token | Percentuale del decode |
|---|---:|---:|
| Embedding e coordinatore, layer `0:12` | 20,99 ms | 26,28% |
| Worker ROCR 0, layer `13:18` | 10,76 ms | 13,47% |
| Worker ROCR 1, layer `19:24` | 11,21 ms | 14,04% |
| Worker ROCR 3, layer `25:30` | 11,08 ms | 13,88% |
| Worker ROCR 4, layer `31:36` | 12,00 ms | 15,03% |
| Worker ROCR 5, layer `37:42` | 11,13 ms | 13,93% |
| Output head | 1,50 ms | 1,87% |
| Conversioni FP16, TCP e scheduling distribuito | 1,11 ms | 1,39% |
| Sampling e gestione API | 0,09 ms | 0,11% |
| **Totale per token** | **79,86 ms** | **100,00%** |

Il calcolo dei transformer layer occupa complessivamente circa il 96,63% del
tempo di generazione:

- coordinatore: 26,28%;
- cinque worker: 70,35%;
- totale calcolo dei layer: 96,63%.

Le componenti esterne al calcolo dei layer pesano complessivamente circa il
3,37%:

- output head: 1,87%;
- trasporto, conversioni e scheduling: 1,39%;
- sampling e API: 0,11%.

## Throughput

| Misura | Risultato |
|---|---:|
| Prefill medio | 72,80 token/s |
| Decode con profiler e telemetria | 12,52 token/s |
| Decode normale senza telemetria | circa 14,33 token/s |
| Overhead osservato del profiling | circa 12,6% |

Il profiling genera una riga diagnostica per ogni worker e per ogni token. I
valori assoluti del decode risultano pertanto inferiori al funzionamento
normale, mentre la distribuzione percentuale rimane utile per identificare gli
stadi dominanti.

## Distanza dalla capacità teorica delle GPU

Il fatto che circa il 96,63% del decode sia trascorso nei transformer layer non
significa che le unità di calcolo delle GPU siano sature. La voce `eval`
comprende anche letture HBM, dequantizzazione IQ2/Q2, routing degli esperti,
riduzioni, sincronizzazioni e tempi di lancio dei kernel.

ROCm espone sei GPU Vega 20 `gfx906`, tutte con 60 compute unit wave64:

| GPU | Quantità | CU per GPU | Clock massimo | FP16 teorico per GPU |
|---|---:|---:|---:|---:|
| Radeon VII | 2 | 60 | 1.801 MHz | circa 27,66 TFLOPS |
| Radeon Pro VII | 3 | 60 | 1.700 MHz | circa 26,11 TFLOPS |
| GPU da 32 GB usata come coordinatore | 1 | 60 | 1.730 MHz | circa 26,57 TFLOPS |
| **Totale** | **6** | **360** | — | **circa 160,24 TFLOPS** |

Il picco FP32 aggregato è circa 80,12 TFLOPS. Ogni scheda dispone inoltre di
circa 1 TB/s di banda HBM2, per un massimo nominale aggregato vicino a 6 TB/s.
I riferimenti AMD sono disponibili per
[Radeon VII](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-vega-series/amd-radeon-vii.html),
[Radeon Pro VII](https://www.amd.com/en/products/specifications/professional-graphics.html)
e [Instinct MI50](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50.html).

### Lavoro attivo del modello MoE

Il GGUF descrive 284,33 miliardi di parametri logici, ma il modello seleziona
soltanto 6 dei 256 esperti routed per layer. Il lavoro minimo attivo per token
può essere stimato così:

| Componente | Parametri logici attivi per token |
|---|---:|
| Pesi non-routed sempre utilizzati | circa 7,30 miliardi |
| Sei esperti routed selezionati | circa 6,49 miliardi |
| **Totale attivo** | **circa 13,80 miliardi** |

Contando una moltiplicazione e un'addizione per peso, i matvec rappresentano
circa 27,6 GFLOP utili per token. Attenzione, normalizzazioni e riduzioni
aggiungono altro lavoro, ma non cambiano l'ordine di grandezza del confronto.

### Utilizzo equivalente del picco FP16

| Indicatore | Prefill | Decode |
|---|---:|---:|
| Throughput normale | 72,80 token/s | circa 14,33 token/s |
| Calcolo utile equivalente | circa 2,01 TFLOPS | circa 0,40 TFLOPS |
| Percentuale dei 160,24 TFLOPS aggregati | **circa 1,25%** | **circa 0,25%** |
| Picco disponibile considerando la topologia | circa 91 TFLOPS | circa 26,7 TFLOPS |
| Utilizzo rispetto al picco disponibile | **circa 2,2%** | **circa 1,5%** |
| Distanza dal picco disponibile | **circa 45 volte** | **circa 67 volte** |

Nel prefill la pipeline permette alle GPU di sovrapporre il lavoro, ma lo
sbilanciamento tra i 13 layer del coordinatore e i 6 layer di ogni worker lascia
i worker occupati soltanto per circa il 48-49% del tempo dello stadio più lento.
La capacità teorica realmente disponibile, pesata per questi periodi di
attività, è quindi circa 91 TFLOPS invece dei 160,24 TFLOPS nominali.

Nel decode di una singola sessione le slice devono essere attraversate in
sequenza. In ogni istante lavora principalmente una sola GPU, quindi il tetto
istantaneo pertinente è circa 26-27 TFLOPS, non la somma da 160,24 TFLOPS. La
percentuale aggregata dello 0,25% descrive l'utilizzo dell'intero sistema, ma
l'1,5% rispetto alla GPU attiva è il confronto più significativo.

Queste percentuali sono espresse in FLOP utili del modello. Non misurano
direttamente quante istruzioni intere o FP16 esegue l'hardware durante la
dequantizzazione e non devono essere interpretate come un possibile speedup
automatico di 45-67 volte.

### Utilizzo equivalente della banda HBM nel decode

Dal contenuto del GGUF risultano circa 8,20 GiB di pesi non-routed e 72,56 GiB
di pesi routed. Se vengono letti soltanto i sei esperti selezionati, il minimo
di pesi compressi da attraversare per token è:

```text
8,20 GiB + 72,56 GiB * 6 / 256 = circa 9,90 GiB/token
```

A 14,33 token/s questo equivale a circa 142 GiB/s. Poiché durante il decode
single-stream è attiva principalmente una GPU alla volta, il confronto utile è
con circa 954 GiB/s della singola scheda, non con i 6 TB/s aggregati.

| Indicatore decode | Valore |
|---|---:|
| Banda minima utile stimata | circa 142 GiB/s |
| Banda teorica della GPU attiva | circa 954 GiB/s |
| Percentuale del picco HBM | **circa 15%** |
| Distanza dal limite HBM ideale | **circa 6,7 volte** |
| Roofline puramente teorico | circa 94-96 token/s |

Il roofline da 94-96 token/s è un limite matematico non raggiungibile nella
pratica: presuppone il 100% della banda HBM e nessun costo per dequantizzazione,
routing, cache, attenzione, sincronizzazione o lancio dei kernel.

### Stiamo sfruttando tutta la capacità computazionale?

**No.** I dati disponibili mostrano che né il prefill né il decode sono vicini
alla capacità teorica delle GPU:

- il prefill realizza circa il 2,2% dei FLOP utili rispetto alla capacità
  teorica pesata per il tempo in cui gli stadi possono essere attivi;
- il decode realizza circa l'1,5% dei FLOP utili della GPU attiva;
- il decode utilizza circa il 15% della banda HBM teorica;
- la pipeline single-stream impedisce al decode di utilizzare
  contemporaneamente tutte e sei le GPU;
- il coordinatore sbilanciato limita il parallelismo del prefill.

Il modello MoE riduce molto il numero di esperti calcolati, ma non spiega da
solo l'efficienza osservata: rimangono circa 13,80 miliardi di parametri attivi
e circa 9,90 GiB di pesi compressi da leggere per token. I risultati indicano
quindi margine nei kernel gfx906, nella fusione delle operazioni, nel numero di
launch, nell'accesso agli esperti e nell'orchestrazione della pipeline.

Per attribuire con precisione la perdita residua servono contatori hardware
ROCm, in particolare:

- banda HBM effettivamente letta e scritta;
- occupazione delle compute unit e wave attive;
- utilizzo VALU/FMA e istruzioni intere di dequantizzazione;
- hit rate delle cache L1/L2;
- durata aggregata per famiglia di kernel MoE, attention e quantized matvec;
- tempo tra un kernel e il successivo.

Senza questi contatori è possibile dimostrare che la capacità non è sfruttata
completamente, ma non separare ancora con esattezza quanta parte del limite
derivi da HBM, dequantizzazione, occupazione, kernel launch o dipendenze della
pipeline.

## Colli di bottiglia osservati

1. **Prefill sbilanciato sul coordinatore.** I 13 layer locali richiedono circa
   il doppio del tempo delle slice da 6 layer assegnate ai worker.
2. **Decode dominato dal calcolo GPU.** Il 96,63% del tempo è trascorso nei
   transformer layer; ottimizzare TCP o la finestra distribuita può produrre
   soltanto miglioramenti limitati nel decode.
3. **Worker `31:36` leggermente più lento.** La media è 12,00 ms per token e
   sono stati osservati picchi di circa 15,2 ms. Conviene verificare clock,
   temperatura e power cap della GPU corrispondente a ROCR 4.
4. **Costo della rete contenuto.** Trasporto e orchestrazione pesano circa
   l'1,39% nel decode; i trasferimenti tra worker valgono circa lo 0,9% del
   lavoro seriale del prefill.

## Analisi della regressione Q8 nel decode

Il 29 luglio 2026 è stato eseguito un confronto controllato per spiegare il
calo osservato da circa 12-14 token/s a circa 8 token/s. Il test usa sempre lo
stesso GGUF, lo split bilanciato `0:7 + 5x7`, `ctx=32768`, chunk 64, finestra 5,
un prompt fisso da 3.105 token e 128 token greedy. Anche l'output head resta sul
coordinatore in tutte le varianti.

| Variante | Prefill | Decode 128 token | Throughput decode |
|---|---:|---:|---:|
| Commit storico `d2b5bea`, prima della regressione | 32,49 s | 10,49 s | 12,20 token/s |
| Branch corrente senza correzione | 31,53 s | 16,00 s | 8,00 token/s |
| Correzione Q8 gfx906, esecuzione 1 | 31,55 s | 10,41 s | 12,29 token/s |
| Correzione Q8 gfx906, esecuzione 2 | 31,52 s | 10,41 s | 12,29 token/s |
| Correzione Q8 gfx906 bit-exact finale | 31,50 s | 10,40 s | **12,31 token/s** |

Rispetto allo stesso binario con la correzione disattivata, il throughput sale
da 7,99 a 12,31 token/s, cioè **+54,1%**. La latenza di generazione scende da
125,1 a 81,2 ms/token, cioè **-35,1%**. Il tempo di prefill cambia meno dello
0,1% ed è quindi indistinguibile dal rumore della misura. La correzione recupera
interamente la prestazione del commit storico equivalente (+0,9%).

### Causa isolata

Il confronto del codice individua la regressione nel commit upstream
`ef8d923` (`Add ROCm GLM 5.2 support`). Quel commit ha rimosso il dispatch
one-token che quantizzava una sola volta l'attivazione in Q8 e riutilizzava il
risultato per tutte le righe di output. I kernel prequantizzati sono rimasti nel
backend, ma il decode è passato sempre ai kernel con input F32. Questo percorso
è più preciso in linea teorica, ma su Vega 20 `gfx906` ripete molto più lavoro
per ogni riga, è sensibilmente più lento e, nel test asincrono esteso descritto
sotto, espone anche una race numerica.

La prova causale usa un singolo binario. Con
`DS4_ROCM_DISABLE_Q8_PREQUANT_DECODE=1` lo stesso eseguibile torna a 7,99
token/s e produce lo stesso JSON del branch non corretto; rimuovendo la
variabile torna a 12,29-12,31 token/s. Non sono quindi responsabili né la
posizione dell'output head, né lo split dei layer, né differenze di build.

### Profilo interno prima e dopo

Il profiler sincrono aggiunge overhead; i tempi seguenti sono la somma di
quattro token e vanno usati per confrontare gli stadi, non come throughput
normale. Le percentuali sono normalizzate separatamente sul totale interno di
ciascuna variante.

| Stadio | Prima | % prima | Dopo | % dopo | Variazione tempo |
|---|---:|---:|---:|---:|---:|
| Attention output Q8 | 117,80 ms | 22,84% | 34,99 ms | 9,86% | **-70,3%** |
| Q path | 78,14 ms | 15,15% | 24,13 ms | 6,80% | **-69,1%** |
| Shared expert gate/up | 37,79 ms | 7,33% | 13,85 ms | 3,90% | **-63,4%** |
| Shared expert down | 19,61 ms | 3,80% | 8,52 ms | 2,40% | **-56,5%** |
| Routed MoE | 59,86 ms | 11,61% | 62,42 ms | 17,59% | +4,3% |
| Inverse RoPE/attention | 62,16 ms | 12,05% | 63,69 ms | 17,94% | +2,5% |
| **Totale stadi interni** | **515,68 ms** | **100,00%** | **354,93 ms** | **100,00%** | **-31,2%** |

I quattro stadi Q8 interessati scendono complessivamente da 253,34 a 81,48 ms,
cioè **-67,8%**. Routed MoE, attenzione vera e propria, HC e router rimangono
sostanzialmente invariati: la piccola oscillazione del profilo a quattro token
non è correlata al recupero di throughput. Questo conferma che il collo di
bottiglia regressivo non era la selezione o il calcolo degli esperti.

### Forma della correzione e controlli di qualità

La modifica è confinata al backend ROCm e attivata a compile time soltanto con
`DS4_GFX906`; le altre architetture conservano il dispatch upstream. Inoltre:

- il percorso prequantizzato resta attivo anche con `--quality`: il test esteso
  ha mostrato che il percorso F32 asincrono può corrompere i logits su gfx906;
- `DS4_ROCM_DISABLE_Q8_PREQUANT_DECODE=1` fornisce un rollback esclusivamente
  diagnostico; per confronti numerici affidabili va usato con sincronizzazione
  dei kernel (`AMD_SERIALIZE_KERNEL=3`);
- la proiezione Q8 doppia quantizza l'attivazione una volta, ma usa due kernel
  row-wise identici alle proiezioni singole, rispettando il contratto bit-exact;
- la suite `--metal-kernels`, eseguita attraverso il backend ROCm gfx906, passa;
- `make rocm-regression ROCM_ARCH=gfx906` passa, inclusi long-context smoke,
  attenzione di riferimento, shim WMMA, packing Q8 wave64 e router wave64;
- un vettore ufficiale model-backed (`short_code_completion`) passa sia con il
  percorso ottimizzato sia con il rollback upstream, usando streaming SSD su
  una singola GPU gfx906.

Il percorso veloce quantizza le attivazioni Q8 e quindi non promette identità
numerica con il percorso F32; una generazione lunga può divergere nei token per
normale sensibilità autoregressiva. Il primo campione eseguito con streaming SSD
è stato scartato: su prompt così brevi la prefill decode-style non era
ripetibile. Il gate definitivo usa invece la stessa pipeline residente sulle
sei gfx906 del benchmark (`0:7 + 5x7`, output sul worker finale), chunk 64,
finestra 5 e attivazioni distribuite F16.

La variante prequantizzata è stata eseguita due volte sulle prime 10
continuazioni e i TSV risultanti sono **identici byte per byte**. Ha poi
completato tutti i 100 casi ufficiali Flash, per 2.289 token target:

| Metrica suite completa Q8 prequant | Valore |
|---|---:|
| Casi completati | **100/100** |
| Token target | 2.289 |
| NLL media | 0,367134 |
| Primo token uguale all'ufficiale | 67/100 |
| Top-1 API agreement | 86,151% |
| Greedy longest common prefix medio | 6,55 token |
| Top-N recall API | 32,075% |
| Pairwise ranking agreement API | 98,862% |

Il percorso F32 upstream, eseguito senza sincronizzazioni artificiali, fallisce
in modo deterministico in `case_029` al token target 12 perché l'intero vettore
dei logits diventa non finito. Lo stesso caso fallisce sia dopo i primi 29 casi
sia isolato in una sessione nuova. Inserire un dump sincrono a ogni layer oppure
impostare `AMD_SERIALIZE_KERNEL=3` lo fa passare: il comportamento identifica
una race asincrona gfx906, non una variazione statistica del modello. Di
conseguenza non esiste un risultato upstream valido sui 100 casi con cui fare
un A/B completo non sincronizzato.

Per separare la qualità dalla race è stato eseguito un confronto A/B
sincronizzato sulle prime 10 continuazioni, 240 token totali. La sincronizzazione
è applicata soltanto al rollback F32; la variante prequantizzata usa il percorso
asincrono normale e ripetibile.

| Metrica qualità | F32 sincronizzato | Q8 prequant gfx906 | Differenza |
|---|---:|---:|---:|
| NLL media | 0,363771 | 0,358652 | **-0,005119 (-1,41%)** |
| Perplexity equivalente | 1,4387 | 1,4314 | **-0,51%** |
| Casi vinti | 5 | 5 | parità |
| Primo token uguale all'ufficiale | 7/10 | 7/10 | invariato |
| Top-1 API agreement | 85,833% | 85,417% | -0,417 punti (1/240) |
| Greedy longest common prefix medio | 7,4 token | 7,4 token | invariato |
| Top-N recall API | 34,990% | 34,990% | invariato |
| Pairwise ranking agreement API | 99,217% | 99,086% | -0,131 punti |

Non emerge una perdita di qualità: la NLL e la perplexity migliorano
leggermente, primo token e LCP restano invariati e la differenza top-1 riguarda
un solo token su 240. Insieme al completamento deterministico della suite da
100 casi, il risultato rende il percorso prequantizzato un miglioramento sia di
throughput sia di robustezza su gfx906.

## Limiti della misura

Queste percentuali descrivono la modalità residente e una posizione di
contesto di circa 2.400 token. Con un contesto già riempito da centinaia di
migliaia di token, il costo dell'attenzione aumenta e la ripartizione può
cambiare.

La modalità di `run-context.sh` non è rappresentata da queste tabelle. Quando
si abilita lo streaming SSD degli esperti, lettura, caricamento e gestione della
cache degli esperti diventano componenti importanti e richiedono un profiling
separato.

## Ottimizzazione della ripartizione per il profilo velocità

Il profilo originale usava 13 layer sul coordinatore e 6 layer su ciascun
worker. A `ctx=32768` è possibile mantenere sette layer residenti su ogni GPU da
16 GB e usare una ripartizione più bilanciata:

```text
coordinatore: 0:7
worker:       8:14, 15:21, 22:28, 29:35, 36:42
```

Il confronto è stato eseguito con lo stesso prompt da 2.418 token, 64 token di
output, chunk 64, finestra 5 e telemetria attiva.

| Misura | Split 13 + 6x5 | Split 8 + 7x5 | Variazione |
|---|---:|---:|---:|
| Prefill | 72,80 token/s | 109,28 token/s | **+50,1%** |
| Tempo prefill | 33,22 s | 22,13 s | **-33,4%** |
| Decode profilato | 12,52 token/s | 12,61 token/s | **+0,7%** |
| Tempo totale 2.418+64 | 38,33 s | 27,21 s | **-29,0%** |

La modifica risolve il principale sbilanciamento del prefill, ma non accelera
in modo sostanziale il decode, che continua ad attraversare le sei slice in
sequenza. Per questo `run-speed.sh` usa lo split 8 + 7x5 a 32K, mentre
`run-context.sh` conserva lo split 13 + 6x5 per lasciare spazio alla KV cache da
1M token e allo streaming SSD.

## Valutazione preliminare del tensor parallelism

RCCL 2.27 è installato, ma l'inizializzazione TP2 sulle coppie PCIe non è
utilizzabile con l'attuale configurazione di boot. RCCL segnala l'assenza di
`iommu=pt` come possibile causa di instabilità o blocco e il test P2P si arresta
durante l'inizializzazione. Anche il fallback con P2P disabilitato non è
affidabile e ha terminato con `SIGSEGV`.

Il tensor parallelism non viene quindi integrato nel percorso di produzione
finché il sistema non viene avviato con una configurazione IOMMU compatibile e
un microbenchmark RCCL TP2 non completa correttamente. Il PP6 bilanciato offre
nel frattempo un miglioramento verificato senza modificare kernel, formato dei
pesi o semantica del modello.
