# Notizen zur Bachelorarbeit

Arbeitsnotizen zu Entscheidungen, Messwerten und offenen Punkten.
Basis fürs Methodikkapitel und zur Vorbereitung aufs Kolloquium.

## Aufbau

- `config.py` — alle Parameter zentral, keine Logik. Alle Module importieren hieraus, nie umgekehrt.
- `data.py` — lädt die 10kGNAD-CSVs, zieht Teilmengen. Kein Preprocessing.
- `classic.py` — klassischer Ansatz: Lemmatisierung (einmalig, gecacht), TF-IDF + logistische Regression (pro Lauf).
- `bert.py` — BERT-Ansatz (offen)
- `experiment.py` — steuert die Läufe, berechnet Metriken, schreibt Ergebnisse (offen)

## Entscheidungen

### Daten

- **CSV wird zeilenweise selbst eingelesen**, nicht mit `pd.read_csv`. Die Datei ist unquotiert und Artikeltexte enthalten selbst Semikolons — nur der Split am *ersten* Semikolon trennt zuverlässig Kategorie und Text.
- **Vorgegebener Train/Test-Split wird übernommen** (9.245 / 1.028). Kein eigener Split, damit die Ergebnisse mit bestehenden 10kGNAD-Benchmarks vergleichbar bleiben.
- **Teilmengen stratifiziert** (`groupby("category").sample()`), damit die Kategorienverteilung erhalten bleibt. Ohne Stratifizierung würde bei kleinen Anteilen der Kultur-Anteil (nur 5 % des Korpus) zufällig schwanken und Streuung erzeugen, die nichts mit den Modellen zu tun hat.
- **Teilmengen werden pro Stufe unabhängig gezogen**, nicht verschachtelt. Bei verschachteltem Ziehen würde sich ein unglücklicher Griff bei 10 % in alle größeren Stufen fortpflanzen und die Streuung über die Wiederholungen unterschätzt.
- **Seed pro Wiederholung**, aber nicht pro Ansatz — beide Modelle trainieren in derselben Wiederholung auf identischen Artikeln.

### Preprocessing (nur klassischer Ansatz)

- **HanTa statt spaCy**: schneller, für deutsche Lemmata besser (Komposita, unregelmäßige Formen), und das Werkzeug, das Wartena in seinen Vorlesungsnotebooks verwendet.
- **Lemmatisierung statt Stemming**: legt Wortformen zuverlässiger zusammen (`genannt → nennen`), ohne falsche Zusammenführungen wie `sieben → sieb`. Merkmale bleiben lesbar für die Auswertung.
- **Satzweises Taggen**: HanTa nutzt den Satzkontext zur Wortartbestimmung, und das Lemma hängt von der Wortart ab. `tag_sent` erwartet einen Satz.
- **Keine POS-Filterung**: Die Stopwortliste deckt die Funktionswörter bereits ab. POS-Tagging findet trotzdem statt, es wird nur nicht zum Filtern genutzt.
- **Keine Kleinschreibung im Preprocessing**: Großschreibung ist im Deutschen ein Signal für Nomen und hilft dem Tagger. Kleingeschrieben wird erst im `TfidfVectorizer`.
- **Rohtext bleibt erhalten**: Spalte `text` für BERT, Spalte `lemmas` für die logistische Regression. Beide im selben DataFrame — dadurch liefert ein Teilmengen-Zug automatisch identische Zeilen für beide Ansätze.
- **Ergebnis wird als Parquet gecacht** (`data/processed/`). Achtung: Bei Änderungen am Preprocessing muss die Datei gelöscht werden, sonst werden die alten Lemmata geladen.

### Klassischer Ansatz

- **TF-IDF wird pro Lauf neu gefittet** — `fit_transform` nur auf der Trainingsteilmenge, `transform` auf dem Testset. Würde man einmal vorab auf allem fitten, hätte auch das 10-%-Modell das volle Vokabular inklusive Testdaten, und die Lernkurve wäre verfälscht.
- **Multinomiale logistische Regression**: Bei neun Klassen wählt sklearn automatisch Softmax statt One-vs-Rest. Ein Modell mit neun Gewichtsvektoren, `coef_` hat die Form (9, n_features).
- **Parameter** (siehe `config.py`): `MIN_DF=1`, `MAX_FEATURES=None` — kein Vokabularfilter, damit die Vokabulargröße allein von der Datenmenge abhängt. `C=1.0` — Regularisierung bewusst aktiv (in Wartenas Notebooks steht `C=1e9`, das ist didaktisch gemeint und würde bei wenigen Artikeln überanpassen). `MAX_ITER=1000` — Default 100 konvergiert bei TF-IDF-Daten oft nicht.

## Messwerte

**Lemmatisierung** (einmalig): ca. 17 min für Train, 2 min für Test. Danach aus dem Cache in Sekunden.

**Klassischer Lauf**: 3 s bei 10 % der Trainingsdaten, 16–17 s bei 100 %.

## Erste Ergebnisse (klassischer Ansatz)

| Trainingsdaten | Seed | Accuracy |
|---|---|---|
| 10 % (924) | 1 | 68,1 % |
| 10 % (924) | 2 | 67,1 % |
| 10 % (924) | 3 | 66,7 % |
| 100 % (9.245) | 1 | 84,9 % |

Streuung über die Seeds: ca. 1,4 Prozentpunkte bei 10 %. Fünf Wiederholungen pro Stufe reichen damit deutlich.

**Zusatzbeobachtung** (nur ein Seed, nicht belastbar): ohne Lemmatisierung 66,9 % bei 10 % und 84,8 % bei 100 %. Preprocessing scheint bei wenig Daten mehr zu bringen als bei viel — der Unterschied liegt aber in der Größenordnung der Seed-Streuung. Für eine belastbare Aussage müsste man beide Varianten über mehrere Seeds mitteln.


### BERT-Ansatz

- **Modell: `deepset/gbert-base`** — speziell auf deutschem Textmaterial trainiert, in Chan/Schweter/Möller (2020) auf 10kGNAD evaluiert und damit als Referenzwert nutzbar.
- **Sequenzlänge 256 Tokens.** Tokenlängen im Trainingsset (gbert-Tokenizer, inkl. Sondertokens): Mittelwert 535, Median 441, Minimum 10, Maximum 4.514.

  | Grenze | Anteil abgeschnittener Artikel |
  |---|---|
  | 128 | 96,2 % |
  | 256 | 76,9 % |
  | 384 | 57,1 % |
  | 512 | 42,2 % |

  Begründung: BERT verarbeitet architekturbedingt höchstens 512 Tokens — selbst dort würden 42 % der Artikel gekürzt. Der Aufwand für Attention wächst quadratisch mit der Sequenzlänge, 256 ist also grob viermal schneller als 512. Da Nachrichtenartikel dem Prinzip der umgekehrten Pyramide folgen (das Thema steht im ersten Absatz), ist die Kürzung für Themenklassifikation vertretbar. Auf CPU ist 512 für zehn Stufen × fünf Wiederholungen nicht durchführbar.



  ## Für die Diskussion

- BERT sieht die Artikel nur gekürzt (bei 256 Tokens 77 %, selbst bei 512 noch 42 % der Artikel), der klassische Ansatz dagegen vollständig. Das ist ein struktureller Nachteil des Sprachmodells in diesem Vergleich, der sich nicht wegoptimieren lässt.


## Messwerte und erste Ergebnisse

Alle Werte mit Seed 1, Testset jeweils vollständig (1.028 Artikel), sofern nicht anders vermerkt.
Einzelmessungen ohne Wiederholungen — nur zur Orientierung, nicht belastbar.

### Accuracy

| Trainingsdaten | Klassisch | BERT |
|---|---|---|
| 1 % (93) | 46,1 % | 47,0 %* |
| 10 % (924) | 68,1 % | 84,0 %* |
| 100 % (9.245) | 84,9 % | — |

\* BERT-Werte gegen 100 Testartikel gemessen, daher nur grob vergleichbar (Unsicherheit rund ±4 Punkte).

Streuung klassisch bei 10 % über Seeds 1–3: 68,1 / 67,1 / 66,7 %.
Klassisch ohne Lemmatisierung: 44,6 % bei 1 %, 66,9 % bei 10 %, 84,8 % bei 100 %.

### Laufzeiten (CPU, lokal)

| Trainingsdaten | Klassisch | BERT |
|---|---|---|
| 1 % (93) | 0,9 s | 244 s |
| 10 % (924) | 3 s | 2.254 s (37,5 min) |
| 100 % (9.245) | 16–17 s | ca. 6 h (hochgerechnet) |

Lemmatisierung einmalig: 17 min (Train), 2 min (Test), danach gecacht.

### Beobachtungen

- **Der Abstand hängt stark von der Datenmenge ab.** Bei 93 Artikeln liegen beide Verfahren praktisch gleichauf (unter 1 Punkt Unterschied), bei 924 Artikeln führt BERT um rund 16 Punkte. Der Bereich, in dem sich das Verhältnis ändert, liegt also zwischen 1 % und 10 % — die Stufen der Lernkurve sollten dort dicht liegen (logarithmische Abstufung statt gleichmäßiger Schritte).
- **Rechenaufwand:** Bei 1 % braucht BERT rund das 270-fache an Rechenzeit für praktisch dieselbe Genauigkeit. Das ist unabhängig vom Schnittpunkt eine Aussage für die Praxis.
- **Lokale Durchführung ist nicht machbar.** Bei zehn Stufen × fünf Wiederholungen käme man auf über 150 Stunden reine Rechenzeit für BERT. Notwendig ist eine GPU (Colab, Kaggle oder Hochschulressourcen).
- Der Trainingsverlust bei BERT fällt sauber (1,95 → 0,33 über drei Epochen), das Fine-Tuning funktioniert also wie erwartet.


### Korrektur zu den BERT-Werten

Die ersten BERT-Messungen liefen gegen nur 100 Testartikel und sind nicht belastbar.
Neu gemessen gegen das vollständige Testset (1.028 Artikel), Seed 1:

| Trainingsdaten | Klassisch | BERT | Laufzeit BERT |
|---|---|---|---|
| 1 % (93) | 46,1 % | 36,1 % | 436 s |
| 10 % (924) | 68,1 % | (läuft) | ca. 40 min |

Bei 1 % liegt der klassische Ansatz **10 Punkte vor BERT**. Der Trainingsverlust
bleibt dort bei 1,945 — reines Raten entspräche ln(9) ≈ 2,197. Bei 93 Artikeln,
Batch-Größe 8 und 3 Epochen sind das nur 36 Optimierungsschritte; der frisch
initialisierte Klassifikationskopf lernt in dieser Zeit kaum etwas.

Das deckt sich mit Li et al. (2022) und Edwards et al. (2021): BERT braucht eine
Mindestmenge an Trainingsschritten und unterliegt darunter einfachen linearen Modellen.

**Offener methodischer Punkt:** Eine feste Epochenzahl bedeutet bei kleinen Teilmengen
sehr wenige Optimierungsschritte. Manche Arbeiten halten stattdessen die Schrittzahl
konstant. Feste Epochen sind der übliche Weg — die Alternative gehört in die Limitationen,
da sonst offen bleibt, ob BERT bei 1 % nur wegen zu weniger Schritte verliert.





## Offene Punkte

- Prozentstufen der Lernkurve festlegen
- Anzahl der Wiederholungen pro Stufe
- BERT-Modell auswählen (gbert-base von deepset oder bert-base-german-cased)
- Sequenzlänge, Batch-Größe, Lernrate, Epochen für BERT
- Anteil der Artikel über 512 Tokens ermitteln (Diskussionspunkt: BERT sieht lange Artikel nur teilweise)
- BERT-Läufe auf CPU zu langsam → Colab