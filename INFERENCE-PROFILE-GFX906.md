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

## Aggiornamento 10 agosto 2026: modello 0731 e ottimizzazione Routed-MoE

Questa sezione fotografa la configurazione corrente e non sostituisce le misure
storiche riportate sopra. Il benchmark usa il GGUF 0731, pesi residenti, sei
processi ROCm (uno per GPU), attivazioni distribuite FP16, finestra 5 e il
seguente split:

```text
ROCR 3, Radeon Pro VII 16 GiB: layer 0:5, coordinatore
ROCR 0, Radeon VII     16 GiB: layer 6:12
ROCR 1, Radeon Pro VII 16 GiB: layer 13:19
ROCR 4, Radeon Pro VII 16 GiB: layer 20:26
ROCR 5, Radeon VII     16 GiB: layer 27:33
ROCR 2, Radeon Graphics 32 GiB: layer 34:42 e output head
```

### Causa del limite di prefill gfx906

Vega 20 non dispone di istruzioni MFMA. La compatibilità rocWMMA implementata
per gfx906 è quindi uno shim software corretto numericamente, ma non un percorso
di accelerazione. Nel prefill Routed-MoE il vecchio dispatch inviava gli esperti
più frequenti proprio a questo shim, rallentando il percorso che avrebbe dovuto
essere veloce.

La correzione mantiene su gfx906 il kernel expert-tiled già disponibile. È
limitata a `DS4_GFX906`; le architetture ROCm con rocWMMA nativo non cambiano.
`DS4_ROCM_ENABLE_EMULATED_MOE_WMMA=1` ripristina il percorso precedente per i
test di regressione.

| Percorso, frontiera 8K | Prefill | Decode | Variazione prefill |
|---|---:|---:|---:|
| rocWMMA emulato precedente | 89,37 token/s | 9,54 token/s | riferimento |
| expert-tiled gfx906 | **159,81 token/s** | 9,54 token/s | **+78,8%** |

Il tempo del prefill diminuisce del 44,1%. Il decode non cambia perché il ramo
hot-expert riguarda il batch di prefill, non il matvec a singolo token.

Il confronto dei logits completi è stato ripetuto su tre prompt da 2.295,
2.697 e 5.355 token. I due percorsi veloci hanno sempre lo stesso argmax e una
sovrapposizione top-20 di 18/20, 18/20 e 19/20. Rispetto al percorso
`--quality`, expert-tiled risulta più vicino in due prompt su tre. Non emerge
quindi una perdita sistematica che giustifichi il costo del rocWMMA emulato.

### Scaling con la lunghezza del contesto

`ds4-bench --repeat-prompt` permette di costruire una frontiera lunga realmente
elaborata, invece di limitarsi ad allocare un KV capiente. La prova seguente
usa una frontiera sintetica di 65.536 token, `ctx-alloc=300000`, chunk 256 e 32
token greedy:

| Frontiera | Prefill | Decode | Variazione rispetto a 8K |
|---|---:|---:|---:|
| 8.192 token | 159,81 token/s | 9,54 token/s | riferimento |
| 65.536 token | **118,18 token/s** | **8,82 token/s** | prefill -26,1%; decode -7,5% |

Il calo non annulla il beneficio Routed-MoE: riflette la quota crescente di
attenzione e accesso al KV con l'aumentare della posizione.

È stata inoltre avviata una frontiera effettiva da 300.000 token con
`ctx-alloc=700000`, senza SSD. I sei processi hanno allocato il contesto,
completato la route e processato il prefill per oltre 13 minuti senza OOM,
disconnessioni o errori numerici. La misura di throughput è stata interrotta
volontariamente quando un sensore GPU ha raggiunto 95 °C; le altre schede erano
tra 77 e 85 °C. Un run completo richiederebbe circa 60-90 minuti nelle attuali
condizioni e sarebbe dominato dal throttling. Il risultato 300K sostenuto va
quindi ripetuto dopo la sostituzione del dissipatore; non viene inventato un
valore estrapolato.

### Profilo distribuito a 64K

Le quote sono calcolate sul solo tempo `eval` dei layer. Nel prefill gli stadi
si sovrappongono in pipeline; nel decode sono seriali. `downstream_wait`
include il lavoro degli stadi successivi e non viene sommato alle quote.

| Layer | GPU | VRAM | Prefill per chunk | Quota prefill | Decode per token | Quota decode |
|---|---|---:|---:|---:|---:|---:|
| 0:5 | Radeon Pro VII | 16 GiB | 1.062,7 ms | 12,3% | 13,37 ms | 12,0% |
| 6:12 | Radeon VII | 16 GiB | 2.102,7 ms | **24,3%** | 18,69 ms | 16,8% |
| 13:19 | Radeon Pro VII | 16 GiB | 1.300,3 ms | 15,0% | 17,62 ms | 15,9% |
| 20:26 | Radeon Pro VII | 16 GiB | 1.407,7 ms | 16,3% | 19,51 ms | 17,6% |
| 27:33 | Radeon VII | 16 GiB | 1.214,4 ms | 14,0% | 17,44 ms | 15,7% |
| 34:42 + output | Radeon Graphics | 32 GiB | 1.557,2 ms | 18,0% | 24,36 ms | **21,9%** |
| **Totale seriale equivalente** | — | — | **8.644,9 ms** | **100,0%** | **110,98 ms** | **100,0%** |

Lo stadio `6:12` limita il prefill e coincide con la GPU che ha mostrato la
temperatura anomala. L'utilizzo della pipeline è 68,5%. Un bilanciamento
perfetto puramente matematico avrebbe il 45,9% di margine, ma non è realizzabile
con layer interi e gli attuali limiti di VRAM.

Con i vincoli misurati (`max=7,7,7,7,7,10` layer), il modello lineare propone
`7,6,7,6,7,10` e stima +16,7% di prefill a 64K. La proposta sposta un layer
dalla GPU termicamente limitata e non viene applicata al launcher: va ripetuto
il profilo dopo il nuovo dissipatore, perché il costo per layer cambierà.

### DSpark con il kernel corretto

Il confronto immediato a GPU già calde usa frontiera 8K, 256 token, chunk 256
e identico split. In questo modo la differenza non viene attribuita a un avvio
più freddo:

| Modalità | Prefill | Decode | Differenza decode |
|---|---:|---:|---:|
| standard | 143,90 token/s | **10,04 token/s** | riferimento |
| DSpark, soglia 0,9 | 138,41 token/s | **6,54 token/s** | **-34,9%** |

Sulle gfx906 il costo di proposta e verifica supera il lavoro evitato anche con
un buon acceptance rate. DSpark resta disponibile per ricerca, ma non deve
essere attivato nel profilo velocità.

Un secondo confronto ha esteso la generazione forzata a 1.024 token:

| Modalità, 1.024 token | Prefill | Decode | Tempo decode stimato |
|---|---:|---:|---:|
| standard | 167,22 token/s | **9,19 token/s** | 111,43 s |
| DSpark, soglia 0,9 | 161,71 token/s | **7,30 token/s** | 140,27 s |

DSpark rimane quindi più lento del 20,6%. In questo run produce 622 token di
draft e ne accetta 614 (98,71%), con draft che arrivano a 5 token. Riduce le
chiamate target da 1.024 a 552, ma spende 19,78 s nella proposta, 74,61 s nella
verifica batch e 0,98 s nei rollback. Il lavoro risparmiato non ripaga questi
costi sulle gfx906.

Il 98,71% non va interpretato come acceptance rappresentativa di 1.024 token
utili: la continuazione DSpark emette il token EOS alla posizione di output 318
e il benchmark prestazionale continua deliberatamente oltre EOS. I successivi
705 token degenerano nell'alternanza dei token 26/67 (`8a8a...`), estremamente
facile da proporre. Anche in questa condizione artificialmente favorevole
DSpark non supera il decode standard. Il run breve da 256 token, interamente
precedente all'EOS, resta il confronto qualitativamente più significativo.

### DSpark al crescere della frontiera di contesto

Il confronto è stato ripetuto a 16.384 token di input effettivi, mantenendo
chunk 256, `ctx-alloc=300000`, finestra 5 e 256 token di output. Un cooldown
controllato ha fatto partire il caso DSpark da 75 °C, senza power cap o modifica
della ventola:

| Frontiera 16K | Prefill | Decode | Tempo decode |
|---|---:|---:|---:|
| standard | 148,97 token/s | **9,34 token/s** | 27,41 s |
| DSpark, soglia 0,9 | 156,93 token/s | **6,23 token/s** | 41,09 s |

DSpark è **33,3% più lento**. Su 253 chiamate speculative, 250 non producono
alcun draft; vengono proposti e accettati soltanto 3 token. La copertura utile
è quindi 3/256, cioè **1,17%**. Il support model impiega 13,72 s, mentre il
rallentamento end-to-end rispetto alla baseline è 13,68 s: la regressione è
quasi interamente spiegata dalla proposta che non riesce a generare draft.

Il profilo isola il costo sulla GPU finale da 32 GiB: lo stadio `34:42+output`
sale da 21,86 a 75,52 ms/token; gli altri cinque stadi restano sostanzialmente
invariati. A questa frontiera la verifica costa appena 0,33 s, perché quasi non
riceve batch speculativi.

È stato completato anche il baseline a 32.768 token: 144,61 token/s di prefill
e 8,79 token/s di decode. Il corrispondente DSpark è stato interrotto durante
il prefill quando la GPU critica ha raggiunto 95 °C, perciò non viene riportato
un numero incompleto. Le modalità 32K e 64K, con cooldown tra i casi e variante
`-only`, restano pronte per il retest dopo la sostituzione del dissipatore.

### CPU ed expert parallelism

Il trasporto delle attivazioni richiede circa 0,15-0,17 ms per hop nel decode,
meno dell'1% del tempo di calcolo complessivo. Tokenizzazione, socket e prefetch
worker sono già eseguiti lato CPU; il prefetch di ricezione a profondità 2 è
attivo. Spostare interi layer sulla CPU aggiungerebbe letture dalla RAM e due
sincronizzazioni PCIe a un percorso seriale, peggiorando la latenza.

Il backend ROCm dispone ora delle primitive `owned expert` necessarie a
filtrare gli esperti locali, conservare gli output dei sei slot e combinarli
nello stesso ordine FP32 del backend CUDA. La regressione GPU prova tutte le 64
possibili assegnazioni home/peer dei sei slot. Queste primitive sono groundwork
e non attivano da sole l'Expert Parallel: `ds4_gpu_init_multi()` accetta ancora
una sola GPU per processo, mentre il launcher gfx906 usa un processo per GPU in
pipeline parallel. Mancano quindi peer-copy/all-reduce fra device nello stesso
engine e il relativo scheduler. Attivare il flag TP senza quel refactoring non
sarebbe una prova prestazionale, ma un percorso non supportato.

Una possibile ricerca futura è assegnare un esperto routed alla CPU in parallelo
ai cinque eseguiti dalla GPU durante il solo decode. Prima di implementarla
servono un kernel ROCm che escluda gli esperti non posseduti e un microbenchmark
che dimostri che l'esperto CPU termina entro la finestra GPU. Senza questo gate,
l'offload CPU rischia di trasformarsi nel nuovo collo di bottiglia.

### File e riproducibilità

- runner: `benchmark-gfx906-0731.sh`;
- analisi: `speed-bench/distributed_profile.py`;
- confronto logits: `speed-bench/compare_logits.py`;
- 8K expert-tiled: `.ds4-benchmarks/gfx906-0731/20260810-210421-chunk256`;
- 8K rocWMMA emulato: `.ds4-benchmarks/gfx906-0731/20260810-210611-legacy-moe-wmma`;
- 64K: `.ds4-benchmarks/gfx906-0731/20260810-211325-long64k`;
- 300K avviato e interrotto per limite termico:
  `.ds4-benchmarks/gfx906-0731/20260810-212700-long300k`;
- DSpark: `.ds4-benchmarks/gfx906-0731/20260810-212309-dspark256`;
- baseline/DSpark da 1.024 token:
  `.ds4-benchmarks/gfx906-0731/20260810-225512-dspark-long`;
- baseline 16K: `.ds4-benchmarks/gfx906-0731/20260810-231826-dspark-context16k`;
- DSpark 16K dopo cooldown:
  `.ds4-benchmarks/gfx906-0731/20260810-232315-dspark-context16k-only`;
- baseline 32K e DSpark interrotto termicamente:
  `.ds4-benchmarks/gfx906-0731/20260810-230626-dspark-context32k`;
- baseline termico abbinato: `.ds4-benchmarks/gfx906-0731/20260810-212511-chunk256`.

## Aggiornamento 11 agosto 2026: graph, MoE decode e KV-cache lunga

I punti seguenti sono stati misurati sullo stesso GGUF 0731 e sullo split
`6+7+7+7+7+9` descritto sopra. Il punto 6 (batching di più richieste) è stato
deliberatamente escluso.

### HIP Graph nel decode

HIP su Vega 20 non può catturare lo stream legacy nullo. La build gfx906 usa
quindi il default stream per-thread e il backend dispone di warm-up, capture,
replay e invalidazione del grafo. La cattura funziona, ma il confronto a
frontiera 1K non mostra un vantaggio ripetibile:

| Percorso | Prefill | Decode |
|---|---:|---:|
| eager | 108,11 token/s | 11,99 token/s |
| HIP Graph | 107,59 token/s | 12,00 token/s |
| variazione Graph | -0,48% | +0,08% |

Il costo di lancio è quindi già una frazione trascurabile del decode. HIP Graph
resta diagnostico e opt-in con `DS4_ROCM_DECODE_GRAPHS=1`; non viene imposto al
profilo di produzione.

### Profiler del Routed-MoE a singolo token

Il profiler inserito nel percorso IQ2/Q2 reale mostra, sul coordinatore e dopo
384 chiamate, 142,611 ms cumulativi nel Routed-MoE:

| Fase Routed-MoE | Tempo cumulativo | Percentuale MoE |
|---|---:|---:|
| quantizzazione Q8 dell'input | 4,163 ms | 2,92% |
| proiezioni gate/up + SwiGLU | 100,810 ms | **70,69%** |
| quantizzazione Q8 intermedia | 4,225 ms | 2,96% |
| proiezione down | 33,413 ms | **23,43%** |
| **Totale** | **142,611 ms** | **100,00%** |

L'oggetto gfx906 contiene istruzioni `v_dot4_i32_i8`: il percorso sfrutta già
il dot-product intero nativo disponibile su Vega 20. Sono stati provati il
caricamento Q8 condiviso fra gate/up e geometrie da 192 thread; nessuno dei due
ha prodotto un vantaggio ripetibile e le modifiche sono state scartate. Il
target utile rimane quindi una fusione più profonda gate/up, non la riduzione
dei due passaggi di quantizzazione che insieme valgono meno del 6% del MoE.

### Esperimenti FP16 sulla KV-cache: esito e rollback di produzione

La prima variante ha conservato tutta la cache di attenzione compressa in
FP16. A 16K misurava 172,46 token/s di prefill e 10,19 token/s di decode,
contro 157,21 e 9,41 token/s in FP32. Il guadagno era reale, ma anche il drift:
su 129.280 logit dopo 2.697 token il valore RMS era 0,786, l'overlap top-20
18/20 e la generazione greedy divergeva al token 21.

Il test dimensionale ha poi isolato la causa: arrotondare a FP16 i primi 448
valori noPE, già passati dal round-trip FP8 del modello, non cambia alcun logit;
arrotondare i 64 valori RoPE riproduce invece quasi tutto il drift. È stata
quindi implementata e verificata una seconda variante mista:

```text
448 valori noPE FP16 + 64 valori RoPE FP32 = 1.152 byte/riga
cache FP32 di riferimento                  = 2.048 byte/riga
```

La regressione isolata dell'attenzione mista ha errore massimo `2,53e-7`. Nel
modello completo il drift scende a RMS 0,0648, con stesso argmax e overlap
top-20 20/20. Non è però bit-identico e la generazione greedy diverge ancora al
token 21. L'A/B di 256 token mostra inoltre un compromesso sfavorevole:

| Cache, frontiera 1K | Prefill | Decode | KV allocata | Token greedy |
|---|---:|---:|---:|---|
| FP32 | 109,30 token/s | **13,11 token/s** | 3,89 GiB | riferimento |
| noPE FP16 + RoPE FP32 | **112,51 token/s** | 12,86 token/s | 2,53 GiB | diverge al 21 |
| variazione | +2,94% | **-1,91%** | -34,96% | non accettata |

È stata provata anche una codifica noPE normalizzata con sette esponenti
potenza-di-due per riga, capace di ricostruire senza perdita i valori prodotti
dal round-trip FP8. Il confronto completo resta però identico alla variante
mista semplice (RMS 0,0648): il drift residuo deriva dal diverso ordine/FMA
generato nei kernel compatti, non dalla precisione dei valori in cache. Una
espansione preventiva in FP32 eliminerebbe il vantaggio e non è stata mantenuta.

Conclusione: la cache compatta resta soltanto un percorso sperimentale; non è
abilitata dai launcher. `run-speed.sh` usa FP32 e 700K token mantenendo lo split
validato `0:5 + 6:12 + 13:19 + 20:26 + 27:33 + 34:42+output`. Cambiare i
confini in `6x5 + 30:42+output` conserva l'argmax sul prompt di controllo, ma
porta l'RMS dei logits a 0,751 e fa divergere la sequenza greedy al token 21;
per questo il riequilibrio più aggressivo è stato scartato.
`run-context.sh` mantiene FP32 a 1M token tramite SSD streaming e una cache
esatta di 156 esperti. Il valore copre due volte il working set routed del
coordinatore da 13 layer; la configurazione iniziale a 72 esperti pianificava
14,37 GiB per worker ma risultava sotto tale soglia sul coordinatore. Il worker
più pesante (`37:42+output`) con 156 esperti pianifica 15,44 GiB e completa
l'allocazione sulla Radeon VII da 16 GiB.
La capacità residente 700K è quindi nuovamente disponibile senza FP16 e senza
SSD streaming; il percorso compatto non è necessario nel launcher di velocità.
La correzione del report per-slice misura, a 700K, 13,73 GiB pianificati sul
coordinator, 14,52-14,96 GiB sui worker da 16 GiB e 19,60 GiB sul worker finale
da 32 GiB. Avvio, route distribuita, creazione sessione e generazione breve sono
riusciti. Il dump completo di 129.280 logits è bit-identico al riferimento
FP32 (`SHA-256 58be30284f82d98cd79d0d95222f5642deff38df6c8eae2c40c4b3837e08b7a0`).

Il benchmark `long64k-700` è stato interrotto senza riportare throughput quando
la GPU termicamente critica ha raggiunto 87 °C edge e 115 °C junction. Aveva
già allocato 700K, costruito la frontiera sintetica da 65.536 token e completato
la route, senza OOM; la prova di riempimento lungo resta rinviata al nuovo
dissipatore. Risultati: `.ds4-benchmarks/gfx906-0731/20260811-102055-logits-f32`
e `.ds4-benchmarks/gfx906-0731/20260811-102251-long64k-700`.

Sono state provate anche geometrie alternative del kernel IQ2 gate/up senza
cambiare l'ordine aritmetico delle singole righe. Tutte producono gli stessi
256 token del riferimento, ma 128 thread scende a 12,71 token/s, 512 thread a
12,99 token/s e otto righe per gruppo a 11,94 token/s, contro 13,02 token/s del
kernel corrente a 256 thread e quattro righe. Anche queste varianti sono state
scartate: la geometria esistente è la migliore fra quelle misurate.

### Stato dell'Expert Parallel ROCm

Sono implementati e compilati:

- filtro/localizzazione delle coppie token-esperto possedute;
- decode owned IQ2/Q2 con conservazione dei sei output per-slot;
- packing esatto 6→4 e combinazione home/peer;
- combinazione batch per il prefill;
- regressione GPU delle 64 mappe di ownership.

Non è ancora corretto dichiarare un throughput EP: l'engine ROCm resta
single-device e il PP6 corrente comunica solo attivazioni fra processi. Il
passo successivo, separato da questi kernel, è rendere multi-device il backend
ROCm e aggiungere peer-copy/all-reduce. Finché ciò non avviene, le primitive
restano inattive e non possono peggiorare il percorso residente normale.

### Risultati riproducibili dell'11 agosto

- eager/Graph: `.ds4-benchmarks/gfx906-0731/20260811-003040-decode-eager` e
  `.ds4-benchmarks/gfx906-0731/20260811-002935-decode-graph`;
- profiler MoE: `.ds4-benchmarks/gfx906-0731/20260811-003515-moe-profile`;
- FP16 completo 1K: `.ds4-benchmarks/gfx906-0731/20260811-074317-decode-f16`;
- FP32 1K: `.ds4-benchmarks/gfx906-0731/20260811-074215-decode-base`;
- FP16 16K: `.ds4-benchmarks/gfx906-0731/20260811-070746-long16k`;
- FP32 16K: `.ds4-benchmarks/gfx906-0731/20260811-071701-long16k-f32`;
- logits FP32/FP16:
  `.ds4-benchmarks/gfx906-0731/20260811-071351-logits-f32` e
  `.ds4-benchmarks/gfx906-0731/20260811-071433-logits-f16`;
- cache mista FP32/FP16 e riferimento abbinato:
  `.ds4-benchmarks/gfx906-0731/20260811-090114-decode-f16` e
  `.ds4-benchmarks/gfx906-0731/20260811-090019-decode-f32`;
- prova FP16 scalata lossless (stesso RMS della cache mista semplice):
  `.ds4-benchmarks/gfx906-0731/20260811-093852-logits-f16`;
- logit FP32 finali, identici bit per bit al riferimento (RMS e massimo 0):
  `.ds4-benchmarks/gfx906-0731/20260811-092919-logits-f32`;
- geometrie IQ2 corrette: `.ds4-benchmarks/gfx906-0731/20260811-091631-moe-iq2-rows4`,
  `.ds4-benchmarks/gfx906-0731/20260811-091724-moe-iq2-rows8`,
  `.ds4-benchmarks/gfx906-0731/20260811-092152-decode-f32` e
  `.ds4-benchmarks/gfx906-0731/20260811-092259-decode-f32`.

## Aggiornamento 12 agosto 2026: bilanciamento dopo il dissipatore

Dopo la sostituzione del dissipatore, il profilo FP32 da 700K completa una
frontiera effettiva da 65.536 token senza throttling critico. Il nuovo split
assegna `7/7/7/7/7/8` layer ai sei stadi e porta la finestra distribuita da 5 a
6, affinché possa essere presente almeno un chunk per stadio. La GPU da 32 GiB
resta finale e possiede gli ultimi otto layer e l'output head.

| Frontiera | Vecchio split, window 5 | Nuovo split, window 6 | Variazione |
|---|---:|---:|---:|
| Prefill 16K | 175,88 token/s | 203,64 token/s | **+15,78%** |
| Decode 16K | 9,93 token/s | 9,95 token/s | +0,20% |
| Prefill 64K, ctx 700K | 156,89 token/s | 183,30 token/s | **+16,83%** |
| Decode 64K, ctx 700K | 8,88 token/s | 8,91 token/s | +0,34% |

Lo split da solo rende uniformi i tempi dei sei stadi ma non aumenta il
throughput: con window 5 il limite a regime diventa la somma dei tempi divisa
per cinque. A 16K misurava infatti 175,81 token/s. Window 6 elimina questo
limite e realizza il guadagno del bilanciamento.

Il settimo layer portava inizialmente il coordinatore a 16,01 GiB pianificati
su una GPU da 15,98 GiB. La route già terminava su un worker `START:output`, ma
il coordinatore conservava comunque circa 0,52 GiB di output head come
fallback. L'opzione generale `--dist-require-worker-output` evita questa copia
solo quando il launcher sa che il worker finale possiede l'head; il
comportamento preesistente resta il default per le invocazioni manuali. Con la
copia rimossa, il profilo 700K completa inizializzazione e benchmark.

I launcher possono ora descrivere topologie simili con una lista ordinata di
device e conteggi di layer. I device accettano indirizzi PCI stabili, risolti a
runtime tramite `rocminfo`, così un riavvio che modifica gli indici ROCr non
cambia l'assegnazione fisica. Il profilo concreto usa:

```text
PIPELINE_LAYER_COUNTS="7 7 7 7 7 8"
```

Gate numerico sul prompt da 2.697 token: tutti i logits sono finiti, argmax
invariato e overlap top-20 19/20 rispetto al precedente split. Rispetto al
percorso `--quality`, il nuovo split FP16 risulta più vicino del precedente
(`RMS 0,487` contro `0,799`) e conserva lo stesso argmax. La continuazione
greedy coincide per 90 token prima della normale divergenza autoregressiva
causata dal diverso confine di trasporto. Il launcher mantiene KV-cache FP32 e
attivazioni inter-stadio FP16; portare soltanto il trasporto a FP32 non migliora
il confronto numerico.

Risultati:

- 16K, window 5: `.ds4-benchmarks/gfx906-0731/20260812-092908-long16k-f32`;
- 16K, window 6: `.ds4-benchmarks/gfx906-0731/20260812-093213-long16k-f32`;
- 64K/700K, window 6: `.ds4-benchmarks/gfx906-0731/20260812-093818-long64k-700`;
- logits configurazione finale, bit-identici alla prima prova del nuovo split:
  `.ds4-benchmarks/gfx906-0731/20260812-094742-logits-f32`.

## Valutazione TP/EP multiprocesso del 12 agosto 2026

Questa prova risponde alla domanda se convenga sostituire il PP6 con tensor
parallelism (TP), oppure accoppiare le GPU in expert parallelism (EP). RCCL
2.27.7 non è stato usato: il primo `all-reduce` termina in errore anche con il
test ufficiale `rccl-tests`, con `iommu=pt` attivo e anche escludendo la GPU da
32 GiB. È stato quindi realizzato un collettivo HIP IPC a stella, coerente con
l'architettura DS4 a un processo per GPU.

Il trasporto è corretto su tutte e sei le gfx906. La prova a sei processi ha
verificato 8 MiB per rank senza errori; pertanto la Radeon da 32 GiB non è
incompatibile con le altre GPU e non serve modificarne il VBIOS. Per un buffer
FP32 da 28 KiB, rappresentativo di un'attivazione da 7.168 elementi, sono stati
misurati:

| Topologia HIP IPC | Latenza accodata | Latenza sincronizzata | Esito |
|---|---:|---:|---:|
| 5 GPU da 16 GiB | 3,80 us | 41,75 us | PASS |
| tutte le 6 GPU | 4,05 us | 55,32 us | PASS |

La latenza accodata misura il lancio del kernel; quella sincronizzata include
le dipendenze peer e rappresenta il costo osservato dal token successivo.

### TP sui blocchi Q8

È stato aggiunto un kernel Q8 con partizione della dimensione K e riduzione
FP32. Il gate numerico a sei shard 7.168→4.096 ha RMS `1,16e-5` e massimo
assoluto `2,10e-5` rispetto alla proiezione non partizionata. La sola parte di
calcolo può scalare idealmente di 5,55x su una GPU, ma nei blocchi reali di
Flash-0731 la proiezione è troppo breve perché il risparmio compensi
l'all-reduce.

| Blocco reale, TP2 | GPU 0–1 | GPU 2–3 | GPU 4–5 |
|---|---:|---:|---:|
| Attention output Q8, 8.192→4.096 | 0,92x | 0,90x | 0,89x |
| Shared expert Q8 completo | 0,78x | 0,84x | 0,51x |

Un valore inferiore a 1,00x è una regressione. Il test dello shared expert
include gate e up 4.096→2.048, SwiGLU, down 2.048→4.096 e un unico all-reduce
finale. I risultati sono quindi più favorevoli al TP rispetto a una riduzione
dopo ogni proiezione, ma restano negativi su tutte le coppie.

Il TP5 non può inoltre conservare il modello e una KV cache da 700K nelle sole
cinque schede da 16 GiB: 80 GiB aggregati sono inferiori ai circa 86,7 GiB dei
pesi, prima ancora di contare cache e buffer. Servirebbero tutte e sei le GPU,
ma il TP6 misurato aggiunge ulteriore latenza.

### EP sui routed expert IQ2/Q2

La prova EP usa le dimensioni e i formati effettivi di Flash-0731: input
4.096, intermedio 2.048, output 4.096, gate/up IQ2_XXS, down Q2_K e sei expert
attivi. Ogni GPU della coppia possiede tre expert, poi viene eseguita una sola
riduzione FP32. Gli slot non posseduti ora vengono azzerati prima della lettura
dei pesi, invece di calcolare inutilmente l'expert 0.

| Routed MoE decode, EP2 | GPU 0–1 | GPU 2–3 | GPU 4–5 |
|---|---:|---:|---:|
| Tempo completo | 0,3206 ms | 0,3169 ms | 0,3188 ms |
| Tempo EP2 | 0,2689 ms | 0,2774 ms | 0,3318 ms |
| Accelerazione | **1,19x** | **1,14x** | 0,96x |

L'output EP è identico al riferimento nella prova sintetica (RMS e massimo
assoluto pari a zero). Il guadagno resta però confinato al routed MoE. Dal
profilo del modello questa fase pesa circa il 22% del tempo di un layer; anche
applicando 1,19x alla coppia migliore, il limite di Amdahl è circa +3,6% sul
layer. La terza coppia è regressiva e passare da PP6 a tre stadi PP con EP2
ridurrebbe anche il parallelismo del prefill. Il beneficio end-to-end previsto
è quindi 0–4%, entro la variabilità del sistema, non un miglioramento
significativo.

### Decisione

Il PP6 `7/7/7/7/7/8`, finestra 6, resta il percorso di produzione. Il
collettivo e i test TP/EP vengono conservati su un branch sperimentale come
gate riproducibile, ma non vengono collegati al grafo del server e non cambiano
`run-speed.sh` o la pull request gfx906. Una futura integrazione sarà sensata
solo se un kernel fuso o una topologia diversa supera questi gate end-to-end,
in particolare `>1,10x` includendo trasporto e sincronizzazione.

I test aggiunti sono:

- `tests/rocm_tp_ipc_star`: collettivo multiprocesso e verifica peer;
- `tests/rocm_tp_q8_projection`: correttezza della partizione K Q8;
- `tests/rocm_tp_q8_ipc_e2e`: shared expert Q8 end-to-end;
- `tests/rocm_ep_iq2_q2_ipc_e2e`: routed MoE IQ2/Q2 end-to-end.

Esempi di esecuzione:

```bash
make tests/rocm_tp_ipc_star tests/rocm_tp_q8_ipc_e2e \
  tests/rocm_ep_iq2_q2_ipc_e2e ROCM_ARCH=gfx906
./tests/rocm_tp_ipc_star --devices 0,1,3,4,5
./tests/rocm_tp_q8_ipc_e2e --devices 0,1
./tests/rocm_ep_iq2_q2_ipc_e2e --devices 0,1
```

## Aggiornamento 12 agosto 2026: kernel decode gfx906

Il profilo `rocprof` sul Routed-MoE Flash-0731 ha mostrato che gate/up occupava
il 72,3% della fase MoE. Il kernel storico assegnava quattro righe a ogni
sottogruppo da otto lane e lanciava solo 96 workgroup, cioè 1,6 workgroup per
CU sulle 60 CU di Vega 20. Il percorso gfx906 assegna ora una riga per
sottogruppo: 384 workgroup e lo stesso identico ordine aritmetico per riga.

| Routed-MoE Flash-0731 | Prima | Dopo | Variazione |
|---|---:|---:|---:|
| Singola gfx906 | 0,320 ms | 0,265 ms | **-17,2% latenza / 1,21x** |
| Decode PP6 a 1K, media A/B | 13,62 tok/s | 13,96 tok/s | **+2,5%** |
| Decode PP6 finale a 1K | — | 14,00 tok/s | — |

La geometria intermedia da due righe misura 0,281 ms e viene scartata. Anche
la riduzione del down-projection da 256 a 128 thread è neutra entro il rumore.
`DS4_ROCM_DISABLE_MOE_IQ2_RPG1=1` ripristina il kernel storico per diagnosi.

### Attention indexed a contesto lungo

Il Q·K indexed leggeva 768 righe da 512 float con accessi distanti tra le lane.
Il nuovo percorso raccoglie una volta le 256 righe raw e le 512 righe top-k in
un buffer temporaneo trasposto di circa 1,5 MiB. Q·K legge così dati coalescenti;
la somma pesata V continua invece a leggere la cache originale, che è il layout
coalescente per quella fase. Le quattro catene FP32 `s0..s3` e il loro ordine
di riduzione restano quelli del kernel precedente.

| Attention 16K, 64 head × 512 | Prima | Dopo | Variazione |
|---|---:|---:|---:|
| Microbenchmark deterministico | 1,347 ms | 0,382 ms | **-71,6% / 3,53x** |
| Decode PP6 A/B | 10,14 tok/s | 12,27 tok/s | **+21,0%** |
| Prefill PP6 A/B | 204,14 tok/s | 203,84 tok/s | -0,15% (rumore) |

Il confronto nello stesso processo contro il percorso storico dà massimo
assoluto `9,31e-10`, RMS `5,78e-11` e 29.080 valori bit-identici su 32.768.
Una prima variante con riduzione cooperativa wave32 era altrettanto veloce, ma
ha prodotto RMS `0,312` e massimo `2,049` sui logits completi: è stata rimossa.
Il rollback diagnostico del percorso accettato è
`DS4_ROCM_DISABLE_ATTENTION_INDEXED_TRANSPOSE=1`.

L'inverse-RoPE indexed è inoltre eseguito nello stesso kernel dopo la somma V.
Il test confronta tutti i 32.768 float con il precedente kernel RoPE separato:
massimo e RMS sono zero, quindi il risultato è bit-identico. L'A/B completo
misura 12,23→12,27 tok/s (+0,33%); il guadagno è piccolo ma coerente nei sei
tempi per-stadio e non modifica il prefill. Il rollback è
`DS4_ROCM_DISABLE_ATTN_INV_ROPE_FUSE=1`.

Risultati riproducibili:

- attention precedente: `.ds4-benchmarks/gfx906-0731/20260812-125829-long16k`;
- transpose A/B: `.ds4-benchmarks/gfx906-0731/20260812-130030-long16k`;
- percorso finale: `.ds4-benchmarks/gfx906-0731/20260812-132455-long16k`;
- inverse-RoPE on/off: `.ds4-benchmarks/gfx906-0731/20260812-133854-long16k`
  e `.ds4-benchmarks/gfx906-0731/20260812-134045-long16k`;
- profilo per fase: `.ds4-benchmarks/gfx906-0731/20260812-132716-graph`.

### Ipotesi distribuite escluse

Il prefill usa già sender, reader e forwarder separati e sovrappone invio TCP e
calcolo GPU. Nel decode 16K ogni worker spende circa 0,12-0,14 ms per invio e il
totale dei cinque hop resta sotto l'1% dei 78-82 ms/token. Il `downstream_wait`
è quasi interamente il calcolo degli stadi successivi, non tempo CPU recuperabile.
La generazione è autoregressiva e non può mettere in pipeline due token target.
Ulteriore threading CPU o una pipeline più profonda non offre quindi un margine
significativo su questa topologia; il prossimo lavoro utile resta nei kernel GPU.

## Aggiornamento 13 agosto 2026: compressori FP16 e down MoE

Un profilo per fase del decode ha mostrato che le proiezioni F16 del
compressore attention e del compressore indexer leggevano due volte lo stesso
input normalizzato da 7.168 float, con due dispatch distinti. Il nuovo kernel
gfx906 esegue insieme le quattro matrici (due coppie), caricando l'input una
sola volta in LDS. L'ordine di accumulo e la riduzione wave32 di ogni riga non
cambiano.

| Decode PP6, modello 0731 | Prima | Dopo | Variazione |
|---|---:|---:|---:|
| 8K prefill + 256 token | 12,57 tok/s | 13,18-13,23 tok/s | **+4,85-5,25%** |
| Prefill dello stesso test | 182,79 tok/s | 182,52-183,10 tok/s | ±0,17% (rumore) |
| Profiler breve 1K | 13,68 tok/s | 14,64 tok/s | **+7,02%** |
| Proiezioni compressori/layer | ~0,348 ms | ~0,155 ms | **-55,5%** |

Il test deterministico usa esattamente le forme 7.168→1.024 e 7.168→256 e
confronta tutti i quattro output: massimo assoluto zero e nessun elemento
diverso. Anche 16 token nel profiler e 256 token nel test lungo coincidono ID
per ID col percorso precedente. L'ottimizzazione è compilata soltanto per
gfx906; `DS4_ROCM_DISABLE_F16_COMPRESSOR_QUAD=1` resta come rollback.

Il down-projection Q2 del Routed-MoE rileggeva da memoria globale i circa
14 KiB delle sei attivazioni intermedie Q8 per ogni blocco di righe. Copiarle
una volta in LDS accelera la sola fase down del 4–6%, ma la fase pesa circa il
28% del MoE e il beneficio end-to-end misurato è circa 0,2%. Il percorso viene
comunque mantenuto perché l'output EP2 resta bit-esatto e non cambia il layout
dei pesi; rollback: `DS4_ROCM_DISABLE_MOE_Q2_DOWN_SHARED_MID=1`.

Tentativi esclusi durante la stessa campagna:

- parallelizzare i sei expert nella down projection: neutro o regressivo;
- fondere i due dot IQ2 gate/up: +23,6% di latenza per pressione registri;
- workgroup gate/up da 128 o 512 thread: nessun guadagno;
- fondere anche il salvataggio degli stati nei quattro matvec: i due kernel
  eliminati valgono troppo poco e il throughput lungo resta 13,22 tok/s; la
  variante è stata rimossa.

Risultati riproducibili:

- riferimento lungo: `.ds4-benchmarks/gfx906-0731/20260813-111835-baseline`;
- percorso finale lungo: `.ds4-benchmarks/gfx906-0731/20260813-083028-baseline`;
- replica finale lunga: `.ds4-benchmarks/gfx906-0731/20260813-123257-baseline`;
- profiler A/B: `.ds4-benchmarks/gfx906-0731/20260813-082953-graph` e
  `.ds4-benchmarks/gfx906-0731/20260813-082038-graph`;
- verifica finale token/profilo:
  `.ds4-benchmarks/gfx906-0731/20260813-113942-graph`.

## Gate DeepSeek autoregressivo del 14 agosto 2026

Tre candidati gfx906 superavano i microbenchmark isolati: split IQ2 gate/up,
weighted-V indexed con 512 thread e workgroup Q8 verso hyper-connection da
otto righe. Il primo A/B PP6 combinato misurava apparentemente 13,76 token/s
contro 13,36 token/s, ma la generazione greedy divergeva al token 17. Il dato
di velocita' e' stato quindi scartato e ogni candidato e' stato isolato sul
modello DeepSeek-V4-Flash reale.

| Candidato isolato, 8K | Esito autoregressivo |
|---|---|
| IQ2 gate/up split | diverge al token 17 |
| weighted-V indexed 512 | output token `0` dal primo passo |
| Q8 -> HC, 8 righe | 32 token identici, ma 256 token degenerano in `0` |

Tutti e tre i commit sono stati revertiti. La ricostruzione finale passa
`make gfx906 -j16` e l'intera `make rocm-regression ROCM_ARCH=gfx906`; il gate
PP6 8K/256 produce 256/256 token identici al riferimento. Le prestazioni finali
sono 197,14 token/s in prefill e 13,30 token/s in decode, contro rispettivamente
197,21 e 13,36 del riferimento: la differenza e' rumore di misura.

Il launcher include ora `decode-verify` (1K/32 token) e `decode-verify8k`
(8K/32 token) per i bisect rapidi. L'accettazione definitiva di un nuovo
default richiede comunque il gate `window6` completo da 8K/256 token, perche'
il candidato Q8 -> HC ha dimostrato che 32 token possono non esporre una
corruzione tardiva.

## Aggiornamento 14 agosto 2026: EP parziale sulle cinque GPU sane

Il TP delle matrici Q8 rimane negativo: dividere attention output o shared
expert non recupera il costo della sincronizzazione a granularita' decode.
L'estensione del gate EP ai 256 expert reali ha invece confermato che la parte
routed IQ2/Q2 scala. Il test accetta da due a sei processi, ownership arbitrarie
dei sei expert attivi e usa una riduzione soltanto verso il proprietario del
layer.

Una matrice completa dei peer ha anche isolato un problema topologico: la GPU
fisica 4, BDF `0000:63:00.0`, impiega 48-64 us per piccole riduzioni
sincronizzate, contro 16-21 us delle altre schede. I trasferimenti grandi verso
la stessa GPU raggiungono comunque 5-7 GiB/s. E' quindi un'anomalia di latenza
small-message, non un limite generale di banda. La scheda da 32 GiB, fisica 2,
funziona correttamente e velocemente pur essendo la variante `sramecc-`;
`AMDGPU_ARCH` con entrambi i target `sramecc+` e `sramecc-` non spiega i
precedenti problemi TP.

### Router reale e collisioni

Una cattura diagnostica temporanea su DeepSeek-V4-Flash ha raccolto 11.008
selezioni (43 layer per 256 token). Sui 36 layer delle cinque GPU sane, la
distribuzione degli expert tra cinque shard contigui produce 0,1941 ms attesi
per il routed MoE, contro 0,2528 ms su una GPU: circa 1,30x. Una partizione
offline basata sulla co-occorrenza, addestrata su meta' token e verificata
sull'altra meta', riduce ancora la latenza EP del 3,2-3,3%. Questo secondo
guadagno e' utile solo dopo aver integrato il parallelismo principale.

### Home-free EP4 con sovrapposizione dello shared expert

Il candidato piu' forte lascia la GPU proprietaria del layer senza expert
routed per quel layer. I quattro peer sani possiedono tutti i 256 shard,
mentre il root esegue lo shared expert Q8 in parallelo:

```text
root:    router -> broadcast 16 KiB -> shared expert --------> reduce -> HC
helper:                          routed shard IQ2/Q2 ---------/
helper:                          routed shard IQ2/Q2 ---------/
helper:                          routed shard IQ2/Q2 ---------/
helper:                          routed shard IQ2/Q2 ---------/
```

Il broadcast HIP IPC di 28 KiB verso quattro peer sani costa 15-20 us. Il gate
integrato include broadcast, shared expert completo sul root, routed experts
sui quattro helper e reduce finale. L'input dei helper viene prima avvelenato,
cosi' la verifica numerica dipende davvero dal broadcast.

| Gate sintetico Flash-0731 | Tempo |
|---|---:|
| Routed + shared sequenziali, una GPU | 0,3264 ms |
| Pattern helper `2/2/1/1` completo | 0,1975-0,2090 ms |
| Media dei nove pattern, pesata per sei scelte su quattro helper | circa 0,222-0,225 ms |
| Accelerazione media | **circa 1,45-1,47x** |

Tutti i nove pattern di collisione, incluso `6/0/0/0`, passano con errore
relativo massimo inferiore a `4e-7`. Anche ciascuna delle cinque GPU sane usata
a turno come root passa; sul pattern `2/2/1/1` i tempi osservati sono
0,202-0,212 ms. Lo shared expert da circa 0,085 ms viene quindi interamente
nascosto dal lavoro degli helper nei pattern dominanti.

Nel profilo del modello reale, routed e shared valgono rispettivamente circa
0,298 e 0,105 ms per layer. Scalando prudentemente il risultato sintetico, il
margine sui 36 layer sani e' stimato in 4,5-6,5 ms per token, ossia circa
4-9% secondo la lunghezza del contesto. Questa e' una proiezione, non ancora
un risultato end-to-end del server.

### Residenza e piano d'integrazione

Il layout puo' conservare lo stesso budget routed attuale. Per gli otto layer
posseduti dalla scheda da 32 GiB, gli altri quattro peer tengono 64 expert
ciascuno. Per i 28 layer posseduti dalle schede da 16 GiB, la scheda da 32 GiB
tiene 73 o 74 expert e gli altri tre peer 60 o 61. I totali restano esattamente
1.792 slot expert-layer per ogni GPU da 16 GiB e 2.048 per quella da 32 GiB.
La GPU fisica 4 mantiene i propri sette layer in PP e non partecipa al gruppo
EP a bassa latenza.

Passare dalla residenza PP alla residenza EP richiede di ricollocare circa
9-10 GiB per GPU. La banda peer misurata stima 0,7-1,5 s se le copie sono
parallele; il break-even e' quindi nell'ordine di 300-700 token generati. Il
prefill deve restare PP6 e il resharing va eseguito soltanto prima di un decode
abbastanza lungo, oppure preparato direttamente dal file modello.

Il runtime distribuito attuale attiva un solo processo per stadio e lascia gli
altri in attesa sulla catena TCP. L'integrazione richiede percio' un servizio
EP persistente per processo, scambio iniziale degli handle HIP IPC e una
barriera per layer. Finche' quel ciclo non passa il gate DeepSeek
autoregressivo completo, il percorso di produzione resta PP6 e
`run-speed.sh` non cambia.

## Benchmark comparativo del 14 agosto 2026

Il costo della semplificazione della parita' dei segni IQ2 e' stato misurato
anche sul modello completo, non soltanto sul kernel isolato. La build corrente
(`a8ee784`, con lo stesso percorso di produzione di `cb9ec7f`) e' stata
confrontata con `fd6d4e6`, cioe' il commit immediatamente precedente alla
modifica IQ2. La parte EP aggiunta da `a8ee784` e' usata soltanto dai test e
non entra nel percorso di esecuzione di `ds4`.

Le due build sono state compilate in directory separate e lanciate in ordine
alternato. Il caso `decode-eager` usa PP6 `7/7/7/7/7/8`, contesto 1.024,
256 token generati, chunk di prefill 256 e HIP Graph disabilitati. Dei cinque
run della versione precedente, uno ha generato 256 token `0`: il suo apparente
15,54 token/s deriva dal calcolo corrotto ed e' escluso dalle medie.

| Metrica | `fd6d4e6`, 4 run validi | corrente, 5 run validi | Delta |
|---|---:|---:|---:|
| Decode medio | 14,7825 tok/s | 14,8760 tok/s | **+0,63%** |
| Decode mediano | 14,78 tok/s | 14,88 tok/s | **+0,68%** |
| Decode steady medio | 14,7975 tok/s | 14,8900 tok/s | **+0,63%** |
| Prefill medio | 112,1625 tok/s | 112,2800 tok/s | +0,10% |
| Primo token mediano | 67,764 ms | 67,132 ms | **-0,93%** |
| Somma eval dei sei stadi/token | 65,9257 ms | 65,5218 ms | **-0,61%** |

Il guadagno e' piccolo ma coerente tra throughput steady e tempo GPU aggregato;
la parita' IQ2 semplificata rimane quindi abilitata. Il prefill e' invariato
entro il rumore, come atteso per una modifica rivolta soprattutto al decode
Routed-MoE.

Questo A/B ha anche esposto che il greedy decode non e' ancora sempre
bit-riproducibile. La sequenza di riferimento compare in 3/5 run di entrambe
le build; la corrente ha due divergenze tardive non nulle, mentre la precedente
ha una divergenza tardiva e il run interamente nullo. I cinque run correnti non
mostrano quindi la corruzione catastrofica, ma il campione non dimostra che la
patch l'abbia eliminata. Il confronto dei token resta un gate obbligatorio e
le prestazioni del run nullo non devono mai essere accettate.

La replica a tre campioni del gate EP sintetico conferma invece il margine del
parallelismo parziale:

| Distribuzione dei sei expert attivi | Sequenziale | EP completo | Speedup |
|---|---:|---:|---:|
| EP2 `3/3` | 0,2530 ms | 0,1908 ms | **1,326x** |
| EP3 `2/2/2` | 0,2528 ms | 0,1852 ms | **1,365x** |
| EP5 `2/1/1/1/1` | 0,2532 ms | 0,1711 ms | **1,480x** |
| Home-free EP4 `0/2/2/1/1`, routed + shared | 0,3265 ms | 0,1981 ms | **1,648x** |

Tutte le repliche EP passano il confronto numerico. Questi dati hanno
giustificato il successivo gate end-to-end, descritto sotto; da soli non
giustificano l'abilitazione del percorso nel server.

Risultati A/B completi:

- corrente: `.ds4-benchmarks/gfx906-0731/20260814-093840-decode-eager`,
  `20260814-094117-decode-eager`, `20260814-094312-decode-eager`,
  `20260814-094608-decode-eager`, `20260814-094748-decode-eager`;
- precedente: `.ds4-benchmarks/gfx906-0731-ab/fd6/` con run
  `20260814-093934-decode-eager`, `20260814-094026-decode-eager`,
  `20260814-094226-decode-eager`, `20260814-094700-decode-eager`,
  `20260814-094841-decode-eager`.

## Gate end-to-end TP/EP del 14 agosto 2026

### Expert parallelism completo: respinto

Il servizio EP persistente e' stato integrato temporaneamente nel decode reale,
con ricollocazione dei pesi PP->EP e dispatch per layer. Il microbenchmark
routed+shared rimane favorevole, ma il modello completo non recupera le
barriere interprocesso ripetute per ogni layer:

| Percorso | Prefill | Decode steady | Decode complessivo |
|---|---:|---:|---:|
| PP6 residente | 19,41 tok/s | circa 15,6 tok/s | **15,83 tok/s** |
| EP persistente dopo transizione | 19,73 tok/s | **15,35 tok/s** | 9,15 tok/s |

La transizione dei pesi richiede 11,306 s e, anche ignorandola, EP e' circa il
3% piu' lento del PP6. L'arena compatta provata per ridurre il costo di
residenza ha inoltre corrotto l'output. L'integrazione di produzione e'
stata rimossa: il risultato sintetico di 1,45-1,65x riguarda soltanto una
porzione che vale circa il 20% del token e non si traduce in un guadagno E2E.

### Sharding della proiezione output Q8: respinto

L'output `7168 -> 129280` e' un caso di tensor parallelism per righe: ogni GPU
calcola vocaboli disgiunti. Il gate process-per-GPU `TP2..TP6` passa il
confronto del candidato top-1; TP5 conserva il 99,7% del guadagno TP6:

| Gate isolato TP5 | Tempo |
|---|---:|
| Output completo su una GPU + top-1 | 2,0538 ms |
| TP5 sincronizzato + top-1 | 0,4807 ms |
| Broadcast di 28 KiB | 0,0239 ms |
| Riduzione del tempo isolato | **1,5731 ms, 4,27x** |

Il server deve pero' conservare tutti i 129.280 logits per sampling e API, non
soltanto il top-1. Il percorso completo con gather e' stato profilato cosi':

| Proiezione reale | Tempo medio/token |
|---|---:|
| Q8 completa sulla GPU finale | **1,0522 ms** |
| TP5: broadcast | 0,0207 ms |
| TP5: dispatch | 0,0129 ms |
| TP5: shard locale | 0,2398 ms |
| TP5: attesa helper | 0,3944 ms |
| TP5: gather logits | 0,3703 ms |
| **TP5 totale** | **1,0380 ms** |

Il risparmio effettivo e' appena 0,0142 ms, circa lo 0,02% di un token da
65-68 ms. Nell'A/B non profilato la generazione passa da 14,85 a 14,79 tok/s;
un secondo A/B profilato ha anche prodotto una divergenza greedy tardiva.
L'integrazione server e' stata rimossa. Rimane il gate riproducibile
`tests/rocm_tp_q8_output_ipc_e2e.cu`, utile se in futuro il protocollo potra'
richiedere esplicitamente solo il top-1.

### Mappa fisica e granularita' di prefill

Anche le alternative senza modifiche ai kernel sono state chiuse con misure:

| Variante, frontiera 8K | Prefill | Decode |
|---|---:|---:|
| Chunk 256 corrente | **196,71 tok/s** | **13,30 tok/s** |
| Chunk 384 | 191,02 tok/s | 13,30 tok/s |
| Chunk 512 | 180,78 tok/s | 13,22 tok/s |
| Coordinatore spostato su ROCR 5, replica migliore | 192,37 tok/s | 13,27 tok/s |

Una nuova prova a frontiera 1K spostando il coordinatore da ROCR 3 a ROCR 1
ha peggiorato il decode da 15,48 a 14,82 tok/s e il prefill da 113,37 a
112,37 tok/s, con divergenza greedy tardiva. La mappa BDF di `run-speed.sh`,
il chunk 256 e PP6 `7/7/7/7/7/8` restano quindi il default misurato.

### Margine residuo concreto

Per una singola sequenza non resta un altro cambio di topologia gia'
dimostrato positivo. Le due aree con peso sufficiente sono i kernel della
singola GPU attiva: Routed-MoE vale circa 0,26-0,30 ms per layer e il blocco
attention/rope circa 0,25 ms per layer a frontiera 1K, crescendo col contesto.
Output sharding, Q8 K-split, EP completo, HIP Graph, chunk piu' grandi e nuove
mappe fisiche sono ora esclusi da misure E2E. Il prossimo candidato deve quindi
superare prima un micro-gate di almeno 5% sulla fase interessata e poi il gate
PP6 8K/256 con token non nulli; guadagni inferiori a circa 0,5 ms/token non sono
distinguibili in modo affidabile dal rumore e non vanno portati in produzione.

Il profiler ora inserisce un confine anche prima della prima fase del layer e
dell'output head; in precedenza la prima voce assorbiva il lavoro pendente del
layer precedente. Sul layer 20 a frontiera 1K il totale attribuito e' 1,597 ms:

| Fase decode/layer | Tempo | Quota |
|---|---:|---:|
| Routed-MoE | 0,258 ms | 16,2% |
| Attention indexed + inverse RoPE | 0,256 ms | 16,0% |
| Proiezione attention output | 0,179 ms | 11,2% |
| Hyper-connection pre-attention | 0,162 ms | 10,2% |
| Hyper-connection pre-FFN | 0,159 ms | 10,0% |
| Quattro proiezioni compressori F16 | 0,154 ms | 9,6% |
| Q path | 0,119 ms | 7,4% |
| Router | 0,087 ms | 5,4% |
| Resto | 0,222 ms | 13,9% |

L'output head separato misura 1,268 ms: proiezione Q8 1,046 ms, HC pre 0,105
ms e tutte le altre operazioni 0,117 ms. Questo profilo corretto conferma che
non esiste una singola fase non ancora ottimizzata abbastanza grande da offrire
un salto analogo al +21% ottenuto dalla transpose attention a contesto lungo.
Il margine realistico dei prossimi kernel e' incrementale e va sommato su 43
layer; l'output head, eseguito una volta, non e' un obiettivo prioritario.
