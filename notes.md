# Notizen zur Bachelorarbeit

Arbeitsnotizen zu Entscheidungen, Messwerten und offenen Punkten.
Basis fürs Methodikkapitel und zur Vorbereitung aufs Kolloquium.

## Aufbau

- `config.py` — alle Parameter zentral, keine Logik. Alle Module importieren hieraus, nie umgekehrt.
- `data.py` — lädt die 10kGNAD-CSVs, zieht Teilmengen. Kein Preprocessing.
- `classic.py` — klassischer Ansatz: Lemmatisierung (einmalig, gecacht), TF-IDF + logistische Regression (pro Lauf).
- `bert.py` — BERT-Ansatz: Tokenisierung und Fine-Tuning über den transformers-Trainer.
- `experiment.py` — steuert die Läufe, berechnet Metriken, schreibt Ergebnisse zeilenweise.
- `analysis.py` — mittelt über die Seeds, erzeugt die Lernkurven.

## Entscheidungen

### Daten

- **CSV wird zeilenweise selbst eingelesen**, nicht mit `pd.read_csv`. Die Datei ist unquotiert und Artikeltexte enthalten selbst Semikolons — nur der Split am *ersten* Semikolon trennt zuverlässig Kategorie und Text.
- **Vorgegebener Train/Test-Split wird übernommen** (9.245 / 1.028). Kein eigener Split, damit die Ergebnisse mit bestehenden 10kGNAD-Benchmarks vergleichbar bleiben.
- **Teilmengen stratifiziert** (`groupby("category").sample()`), damit die Kategorienverteilung erhalten bleibt. Ohne Stratifizierung würde bei kleinen Anteilen der Kultur-Anteil (nur 5 % des Korpus) zufällig schwanken und Streuung erzeugen, die nichts mit den Modellen zu tun hat.
- **Teilmengen werden pro Stufe unabhängig gezogen**, nicht verschachtelt. Bei verschachteltem Ziehen würde sich ein unglücklicher Griff bei einer kleinen Stufe in alle größeren fortpflanzen und die Streuung über die Wiederholungen unterschätzt.
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

- **TF-IDF wird pro Lauf neu gefittet** — `fit_transform` nur auf der Trainingsteilmenge, `transform` auf dem Testset. Würde man einmal vorab auf allem fitten, hätte auch das kleinste Modell das volle Vokabular inklusive Testdaten, und die Lernkurve wäre verfälscht.
- **Multinomiale logistische Regression**: Bei neun Klassen wählt sklearn automatisch Softmax statt One-vs-Rest. Ein Modell mit neun Gewichtsvektoren, `coef_` hat die Form (9, n_features).
- **Parameter** (siehe `config.py`): `MIN_DF=1`, `MAX_FEATURES=None` — kein Vokabularfilter, damit die Vokabulargröße allein von der Datenmenge abhängt. `C=1.0` — Regularisierung bewusst aktiv (in Wartenas Notebooks steht `C=1e9`, das ist didaktisch gemeint und würde bei wenigen Artikeln überanpassen). `MAX_ITER=1000` — Default 100 konvergiert bei TF-IDF-Daten oft nicht.

### BERT-Ansatz

- **Modell: `deepset/gbert-base`** — speziell auf deutschem Textmaterial trainiert, in Chan/Schweter/Möller (2020) auf 10kGNAD evaluiert und damit als Referenzwert nutzbar.
- **Sequenzlänge 256 Tokens.** Tokenlängen im Trainingsset (gbert-Tokenizer, inkl. Sondertokens): Mittelwert 535, Median 441, Minimum 10, Maximum 4.514.

  | Grenze | Anteil abgeschnittener Artikel |
  |---|---|
  | 128 | 96,2 % |
  | 256 | 76,9 % |
  | 384 | 57,1 % |
  | 512 | 42,2 % |

  Begründung: BERT verarbeitet architekturbedingt höchstens 512 Tokens — selbst dort würden 42 % der Artikel gekürzt. Der Aufwand für Attention wächst quadratisch mit der Sequenzlänge, 256 ist also grob viermal schneller als 512. Da Nachrichtenartikel dem Prinzip der umgekehrten Pyramide folgen (das Thema steht im ersten Absatz), ist die Kürzung für Themenklassifikation vertretbar. Auf CPU wäre 512 für den geplanten Umfang nicht durchführbar.

- **transformers-Trainer statt eigener PyTorch-Schleife.** Weniger fehleranfällig und näher am Standard. Der Trainer führt intern dieselben Schritte aus: Forward, Loss, Backward, Optimizer-Step, Gradienten zurücksetzen.
- **Keine Evaluation auf dem Testset während des Trainings** (`eval_strategy="no"`, kein Early Stopping, kein `load_best_model_at_end`). Sonst flösse Testwissen in die Modellauswahl ein und die Accuracy wäre zu optimistisch.
- **Keine Checkpoints** (`save_strategy="no"`) — bei 60 Läufen à rund 400 MB sonst zweistellige Gigabyte.
- **Feste Label-Zuordnung** über die Kategorienliste in der Config, nicht aus dem jeweiligen Subset abgeleitet. Sonst könnten sich die Indizes zwischen Läufen verschieben.
- **Parameter:** `BATCH_SIZE=8`, `EPOCHS=3`, `LEARNING_RATE=2e-5`, `WEIGHT_DECAY=0.01`. Die Lernrate ist der etablierte Standard fürs Fine-Tuning (Sun et al. 2019): Der vortrainierte Teil soll nur leicht angepasst, nicht überschrieben werden.
- **Alle Hyperparameter bleiben über alle Läufe konstant.** Variiert wird ausschließlich die Trainingsdatenmenge — sonst wären Unterschiede nicht mehr eindeutig der Datenmenge zuzuschreiben.

### Experimentdesign

- **Stufen:** 0,5 / 0,6 / 0,7 / 0,8 / 0,9 / 1 / 2 / 3 / 4 / 5 %. Dicht im unteren Bereich, weil die ersten Messungen den Schnittpunkt zwischen 1 % und 10 % vermuten ließen. Obergrenze 5 % wegen der Rechenzeit auf CPU.
- **Drei Seeds pro Stufe.** Die Teilmenge wird einmal pro (Stufe, Seed) gezogen und beiden Ansätzen übergeben — Unterschiede können also nicht von verschiedenen Trainingsartikeln kommen.
- **Ergebnisse werden zeilenweise weggeschrieben**, nicht am Ende gesammelt. Bei stundenlangen Läufen wäre ein Absturz sonst teuer.
- **Metriken zentral in `experiment.py`** berechnet (`accuracy_score`, `f1_score` mit `average="macro"`), damit beide Ansätze identisch bewertet werden. Macro-F1, weil der Datensatz unbalanciert ist und die kleinen Kategorien sonst untergehen.
- **Gemessene Laufzeit** umfasst Training und Vorhersage, bei beiden Ansätzen gleich gehandhabt.

### Darstellung

- **Logarithmische x-Achse.** Die Stufen von 0,5 bis 1 % liegen sehr dicht beieinander und würden sich linear zusammenquetschen. Ticks zeigen Anteil und tatsächliche Artikelzahl.
- **Schattiertes Band** für Mittelwert ± Standardabweichung über die Seeds, statt Fehlerbalken oder Einzelkurven. Die Einzelläufe kommen als Tabelle in den Anhang.

## Ergebnisse (erster vollständiger Lauf, 03.09.2026)

10 Stufen × 3 Seeds × 2 Ansätze, Testset immer vollständig (1.028 Artikel).
Mittelwerte über die drei Seeds, Standardabweichung in Klammern (Prozentpunkte).

| Anteil | Artikel | Klassisch | BERT |
|---|---|---|---|
| 0,5 % | 47 | 33,3 % (1,2) | 28,4 % (4,7) |
| 0,6 % | 55 | 37,6 % (0,8) | 29,7 % (5,0) |
| 0,7 % | 66 | 39,4 % (0,9) | 33,9 % (3,2) |
| 0,8 % | 74 | 42,4 % (0,7) | 37,9 % (6,5) |
| 0,9 % | 83 | 40,6 % (1,8) | 36,0 % (5,7) |
| 1,0 % | 93 | 47,7 % (1,4) | 47,0 % (3,6) |
| 2,0 % | 184 | 54,1 % (1,1) | 56,6 % (3,0) |
| 3,0 % | 276 | 57,8 % (1,1) | 65,4 % (3,7) |
| 4,0 % | 369 | 61,5 % (0,3) | 75,1 % (1,7) |
| 5,0 % | 463 | 62,5 % (0,06) | 79,8 % (3,4) |

Laufzeiten pro Lauf: klassisch 0,7–2,8 s, BERT 417–1.341 s.

Ältere Einzelmessungen (Seed 1, volles Testset) für die größeren Stufen:
klassisch 68,1 % bei 10 % und 84,9 % bei 100 %; BERT 84,1 % bei 10 %.

### Beobachtungen

- **Der Schnittpunkt liegt bei rund 1 % der Trainingsdaten** (93 Artikel, etwa 10 pro Kategorie). Darunter führt der klassische Ansatz um 4–8 Punkte, darüber zieht BERT schnell davon: bei 5 % sind es über 17 Punkte Vorsprung. Das deckt sich mit Li et al. (2022) und Edwards et al. (2021).
- **BERT streut deutlich stärker.** Standardabweichung über die Seeds: BERT 3,0–6,5 Punkte, klassisch 0,06–1,8 Punkte. Bei wenig Daten ist BERT also nicht nur schlechter, sondern auch unzuverlässiger — das Ergebnis hängt stark vom Zufall ab. Zusätzliche Zufallsquelle bei BERT ist die Initialisierung des Klassifikationskopfs.
- **Bei 0,9 % geht die Accuracy bei beiden Verfahren leicht zurück** gegenüber 0,8 %. Rauschen bei kleinen Stichproben, kein Effekt — sollte im Text kurz erwähnt werden.
- **Bei 0,5 % ist der F1-Score beider Verfahren fast identisch** (0,174 gegen 0,168), obwohl die Accuracy 5 Punkte auseinanderliegt. Beide treffen dort im Wesentlichen nur die häufigen Kategorien.
- **Trainingsverlust bei BERT** liegt bei den kleinsten Stufen um 2,0 — reines Raten entspräche ln(9) ≈ 2,197. Bei 18 bis 36 Optimierungsschritten lernt der Klassifikationskopf praktisch nichts. Bei 10 % fällt er dagegen sauber auf 0,33.
- **Rechenaufwand:** Bei 0,5 % braucht BERT rund das 580-fache an Rechenzeit und ist dabei schlechter. Auch bei 10 % steht 3 s gegen 37 Minuten.
- **Zusatzbeobachtung ohne Lemmatisierung** (nur ein Seed, nicht belastbar): 44,6 % bei 1 %, 66,9 % bei 10 %, 84,8 % bei 100 %. Preprocessing scheint bei wenig Daten mehr zu bringen als bei viel, der Unterschied liegt aber in der Größenordnung der Seed-Streuung.

## Für die Diskussion

- BERT sieht die Artikel nur gekürzt (bei 256 Tokens 77 %, selbst bei 512 noch 42 % der Artikel), der klassische Ansatz dagegen vollständig. Ein struktureller Nachteil des Sprachmodells in diesem Vergleich, der sich nicht wegoptimieren lässt.
- Der Rechenaufwand unterscheidet sich um zwei bis drei Größenordnungen. Für die Praxis ist das ein eigenständiges Argument, unabhängig vom Schnittpunkt.
- Die höhere Streuung von BERT bei wenig Daten bedeutet: Ein einzelner Lauf ist dort kaum aussagekräftig.

## Limitationen

- **BERT-Läufe sind nicht bit-genau reproduzierbar.** Derselbe Seed lieferte bei Wiederholung leicht abweichende Werte (28,99 % gegen 30,45 %). Der Seed in den `TrainingArguments` deckt nicht alle Zufallsquellen ab; auf CPU kommen Thread-Reihenfolge und nichtdeterministische Operationen hinzu. Die Stichprobenziehung selbst ist reproduzierbar.
- **Feste Epochenzahl** bedeutet bei kleinen Teilmengen sehr wenige Optimierungsschritte. Alternative wäre eine feste Schrittzahl. Feste Epochen sind der übliche Weg — die Alternative gehört in die Limitationen, sonst bleibt offen, ob BERT bei kleinen Mengen nur wegen zu weniger Schritte verliert.
- **Drei Seeds** sind eine grobe Schätzung der Streuung, gerade bei BERT.

## Offene Punkte

- Große Stufen (10 / 25 / 50 / 100 %) nachrechnen — braucht GPU
- Rechenleistung beschaffen (Colab, Kaggle oder gemietete Instanz)
- Wiederaufnahme nach Abbruch in `experiment.py` ergänzen (vorhandene Kombinationen überspringen)
- Trainingsverlust als zusätzliche Spalte in die Ergebnisdatei
- Tabellenexport für LaTeX in `analysis.py`
- Schreiben: LaTeX-Gerüst aufsetzen
