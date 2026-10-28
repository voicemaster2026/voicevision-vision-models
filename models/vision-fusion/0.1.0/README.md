# Vision fusion 0.1.0

Experimentální model PyTorch spojuje RGB snímek, oblast z vektorové kresby
a český či anglický **přepis** povelu. Zvuk zpracovává samostatné ASR.

## Rozsah

CNN pro celý snímek a masku, CNN pro označený výřez a embedding tokenů vstupují
do společné vrstvy. Tři klasifikační hlavy odhadují návrh operace
`inspect_region`, `count_objects` nebo `measure_dimensions`, syntetický tvar
a jeho barvu. ROI se vypočítá z kresby ve zdrojových souřadnicích. Nevzniká
měření, počet objektů ani výsledek inspekce.

Trénovací data tvoří pouze programově generované kruhy, obdélníky, trojúhelníky
a prázdné oblasti, barvy a původní české/anglické šablony. Neobsahují kamerové
snímky, osobní či zákaznická data ani vendor dokumenty. Nová syntetická data
a šablony jsou CC0-1.0. Generátor je v `tools/train_vision_fusion.py`; manifest
uchovává seed, štítky, povel a hashe obrazových/maskových bajtů každého příkladu.

## Rozdělení a hodnocení

Pevný běh používá 3 200 trénovacích a 800 hodnoticích příkladů, 20 epoch,
Adam a seed 20261023. Hodnoticí sada používá oddělené seedy, nové formulace,
větší tvary a jiné posuny. Neslouží k aktualizaci vah ani výběru checkpointu.
Hodnocení a diagnostika vynechaného snímku/textu nebo chybně přiřazené oblasti
se ukládají v `evaluation.json`. Výsledky platí pouze pro tuto syntetickou sadu.

## API

`voicevisionunited.vision_fusion.propose(image_path, strokes, transcript,
model_root=None, image_id=None, aiappdata_result=None)` přijímá seznam tahů:

```json
[{"points": [[10, 10], [40, 10], [40, 40], [10, 40]], "closed": true}]
```

Souřadnice jsou pixely původního snímku v uložené orientaci. Volitelná šířka
otevřeného tahu `width` je v pixelech. Model byl hodnocen pro uzavřené obdélníkové
oblasti; obecné volné tahy nejsou validovanou distribucí. API zachovává identitu
snímku, SHA-256 snímku, vah a kresby a vrací `proposal_only`. Výstup nikdy
nespustí kamerovou ani aplikační akci. Softmax skóre není kalibrovaná jistota.

## Omezení

Způsobilost na skutečných snímcích, objektech, vadách a fyzických dílech není
ověřená. Klasifikace syntetického tvaru nemá význam obecného porozumění scéně.
Model neumí OCR, ASR, kalibraci, číselné měření, počítání ani řízení kamery.
Pro takové úlohy jsou potřebné samostatné ověřené adaptéry a další data.
Tato verze není automaticky doporučena k nasazení.


## AIappdata parallel evidence (0.1.0)

An optional `aiappdata_result` accepts only the pinned VVU adapter's interface
1.1 result. The adapter validates schema, model/weight revisions, scores,
decision, embedding and preprocessing metadata, then recomputes the exact
normalized tensor digest from the same source image. A mismatch is rejected.
Valid evidence is attached as `upstream_visual_evidence` after inference; it
does not affect VisionFusion inputs, scores or proposal. This does not
authenticate an arbitrary producer or validate production quality. The AIappdata
classifier remains a demo model with uncalibrated scores.
