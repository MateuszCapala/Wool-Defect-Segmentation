## Instrukcja obsługi

### 1. Trening modelu
Aby rozpocząć trening modelu, uruchom skrypt `train.py`:

```bash
python src/train.py
```

- Trenuje model na danych treningowych.
- Zapisuje wytrenowany model w pliku `best_model.pth`.

---

### 2. Ewaluacja modelu
Aby ocenić jakość modelu na zbiorze walidacyjnym, uruchom skrypt `eval.py`:

```bash
python src/eval.py
```

- Oblicza metryki, takie jak F1-Score, IoU, Precision i Recall.
- Wyświetla wyniki w konsoli.

---

### 3. Inferencja (generowanie masek)
Aby wygenerować maski dla danych testowych, uruchom skrypt `inference.py`:

```bash
python src/inference.py
```

- Ładuje model z pliku `best_model.pth`.
- Generuje maski dla obrazów testowych.
- Tworzy archiwum `masks.zip` zawierające wygenerowane maski.

---

### 4. Wizualizacja wyników
Aby stworzyć panele wizualizacyjne z wynikami modelu, uruchom skrypt `visualize_results.py`:

```bash
python src/visualize_results.py
```

- Tworzy panele zawierające obraz wejściowy, maskę prawdziwą i maskę przewidzianą przez model.
- Zapisuje panele w folderze `results_vis`.
