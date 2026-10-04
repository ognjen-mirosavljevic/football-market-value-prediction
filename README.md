# Football Player Market Value Prediction

Machine-learning projekat za procenu tržišne vrednosti fudbalera na osnovu učinka na terenu, karakteristika igrača, lige i klupskog konteksta.

Pipeline spaja statistike za sezonu 2024/25 sa istorijskim Transfermarkt valuacijama. Trenutno se modeluju outfield igrači; golmani su izdvojeni jer zahtevaju drugačije metrike.

## Struktura projekta

```text
football-market-value-prediction/
├── data/
│   ├── raw/                         # izvorni CSV fajlovi
│   └── processed/                   # generisani međurezultati (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_matching.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── matching.py
│   └── features.py
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## Notebook workflow

Notebookovi se pokreću redom:

1. **`01_eda.ipynb`** — kvalitet podataka, duplikati, missing values, GK/outfield split, minuti, početni feature-i i distribucija targeta.
2. **`02_data_matching.ipynb`** — agregacija igrača, `identity_key`, `MainSquad`, sezonske valuacije i exact → normalized → fuzzy → ambiguous matching. Rezultat je `data/processed/final_players.csv` uz audit tabele za prihvaćene i nerešene identitete.
3. **`03_feature_engineering.ipynb`** — log target, availability indikatori, jedan train/test split, raw/per-90 feature-i i reliability shrinkage. Rezultat su `train_features.csv` i `test_features.csv`.
4. **`04_modeling.ipynb`** — modeli A–I, RF/XGBoost feature importance, Model H (klub), Model I (prethodna tržišna vrednost), ablation, error analysis i grafikoni.

Zajednička logika je u `src/`, pa notebookovi ostaju kratki i fokusirani na objašnjenje i rezultate.

## Ključne metodološke odluke

### Agregacija i identitet

Igrač može imati više redova nakon promene kluba. Counting statistike se sabiraju, `MainSquad`, glavna pozicija i liga dolaze iz reda sa najviše minuta, a procenti se ponovo računaju iz ukupnih brojilaca i imenilaca. Poznate same-name/same-age kolizije rešavaju se eksplicitno i auditabilno u `src/data_processing.py`.

### Entity matching

Performance i Transfermarkt podaci nemaju zajednički identifikator. Matching zato koristi četiri faze:

1. jedinstveno exact ime;
2. jedinstveno normalizovano ime;
3. fuzzy ime uz proveru kluba i uzrasta;
4. disambiguation exact imena uz klub i uzrast.

Conservation provere garantuju da se identiteti ne dupliraju niti tiho gube tokom join operacija.

### Reliability-adjusted per 90

Naive per-90 vrednosti mogu biti ekstremne kod malog broja minuta. Zato se stopa skuplja ka pozicionom proseku:

```text
w = Min / (Min + 450)
adjusted = w * player_per90 + (1 - w) * position_baseline
```

Pozicioni baseline se računa samo iz train skupa kako bi se sprečio leakage.

### Target

Tržišne vrednosti su snažno right-skewed, pa modeli predviđaju:

```python
log_market_value = np.log1p(market_value_in_eur)
```

Metrike se prikazuju i na log skali i u evrima.

## Pokretanje

Kreirajte virtuelno okruženje, instalirajte zavisnosti i pokrenite Jupyter:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
jupyter notebook
```

Zatim izvršite notebookove od `01` do `04`. `RUN_TUNING = False` u modeling notebooku koristi fiksne parametre za brzo reprodukovanje; postavite ga na `True` da pokrenete `RandomizedSearchCV` za modele E i G.

## Podaci i generisani fajlovi

Ulazni fajlovi su u `data/raw/`. Fajlovi u `data/processed/` i trenirani modeli su generisani artefakti i nisu verzionisani, osim `.gitkeep` placeholdera.
