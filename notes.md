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

## Offene Punkte

- Prozentstufen der Lernkurve festlegen
- Anzahl der Wiederholungen pro Stufe
- BERT-Modell auswählen (gbert-base von deepset oder bert-base-german-cased)
- Sequenzlänge, Batch-Größe, Lernrate, Epochen für BERT
- Anteil der Artikel über 512 Tokens ermitteln (Diskussionspunkt: BERT sieht lange Artikel nur teilweise)
- BERT-Läufe auf CPU zu langsam → Colab