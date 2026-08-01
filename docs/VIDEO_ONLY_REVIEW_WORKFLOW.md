# Video-only pregled trenera

Ovaj tok rada koristi samo Sony iPhone snimke. Sony je glavna vremenska osa, a iPhone se na nju preslikava preko dva ručno potvrđena trostruka tap-ankera. Obrada je lokalna: izvorni fajlovi se samo čitaju, nikada se ne preimenuju, ne menjaju i ne brišu.

## Preduslovi

Potrebni su:

- Python okruženje sa `ultralytics`, `opencv-python`, `numpy`, `scipy` i `torch`;
- `ffmpeg` i `ffprobe` dostupni u `PATH`;
- lokalni YOLO pose model, na primer `yolo11x-pose.pt` u korenu projekta;
- Apple Silicon/MPS je poželjan za YOLO, ali `cpu` je dozvoljen kada je MPS nedostupan;
- težine moraju već postojati lokalno. Import ne sme da preuzima model sa mreže.

Provera alata:

```bash
command -v ffmpeg
command -v ffprobe
python -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
test -s yolo11x-pose.pt
```

## Preflight

Postavi putanje samo kao promenljive. Ne pravi kopije preko izvora i ne koristi izlazni direktorijum unutar `Downloads` ili memorijske kartice.

```bash
export SONY=/Volumes/Untitled/PRIVATE/M4ROOT/CLIP/C0007.MP4
export IPHONE=/Users/pauldudko/Downloads/IMG_3852.mov
export SESSION=/private/tmp/lionjudo-video-review-session/session
export MODEL=/Users/pauldudko/VSProjects/LionOfJudo/yolo11x-pose.pt

ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate \
  -of json "$SONY"
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate \
  -of json "$IPHONE"
mkdir -p /private/tmp/lionjudo-video-review-session/preflight
```

Pregledaj kratke isečke ili kontakt-tablice oko oba kandidata i preseka povrede. Pre importovanja ručno potvrdi:

1. da prvi anker obuhvata tri stvarna tap-a na početku, pre prva tri tap-a koje trener pominje;
2. da drugi anker obuhvata drugu trostruku sekvencu, približno dva bacanja pre kraja;
3. da je izabrani sportista plav i da se njegov seed okvir jasno vidi u Sony kadru;
4. da je presek povrede na Sony osi posle udarca i klečanja, a pre normalnih statistika.

Primer kratkog preflight isečka, bez izmene izvora:

```bash
ffmpeg -hide_banner -loglevel error -ss 108 -i "$SONY" -t 8 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  /private/tmp/lionjudo-video-review-session/preflight/sony_second.mp4
ffmpeg -hide_banner -loglevel error -ss 128 -i "$IPHONE" -t 8 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  /private/tmp/lionjudo-video-review-session/preflight/iphone_second.mp4
ffprobe -v error -show_entries format=duration,size -of json \
  /private/tmp/lionjudo-video-review-session/preflight/sony_second.mp4
```

## Import i potvrde

Import zahteva dve potvrđene `AnchorPair` vrednosti. `sony_s` i `iphone_s` su sekunde na odgovarajućim fajlovima, `user_confirmed` mora biti `true`, a `triple_tap_count` tačno `3`. Primer poziva iz korena projekta:

```bash
python - <<'PY'
from pathlib import Path
from pipeline.video_review_contract import AnchorPair
from pipeline.video_review_import import import_session

import_session(
    sony=Path("/Volumes/Untitled/PRIVATE/M4ROOT/CLIP/C0007.MP4"),
    iphone=Path("/Users/pauldudko/Downloads/IMG_3852.mov"),
    output_dir=Path("/private/tmp/lionjudo-video-review-session/session"),
    anchors=[
        AnchorPair("pocetak", sony_s=10.000, iphone_s=30.000, user_confirmed=True, triple_tap_count=3),
        AnchorPair("kontrola", sony_s=110.863, iphone_s=130.359, user_confirmed=True, triple_tap_count=3),
    ],
    injury_cutoff_s=126.000,
    blue_seed=(x, y, x + w, y + h),
)
PY
```

Pošto importer učitava podrazumevanu težinu `yolo11x-pose.pt` iz trenutnog direktorijuma, a težina može biti van projekta, pripremi lokalni link pre gornjeg poziva:

```bash
ln -s "$MODEL" yolo11x-pose.pt
```

U gornjem primeru zameni `x, y, w, h` koordinatama potvrđenim na jasnom Sony kadru; importeru se prosleđuju izvedeni uglovi `(x1, y1, x2, y2)`, dok se u izveštaju beleži `(x, y, w, h)`. Link mora pokazivati na postojeću lokalnu težinu; ne dozvoli automatsko preuzimanje. Import pravi `review.json`, analize, preview fajlove, klipove događaja i `media/session_side_by_side.mp4`. Ne pokreći import preko već anotirane sesije; za novu obradu koristi prazan direktorijum. `force_reimport=True` je dozvoljen samo kada je očuvanje postojeće trenerove beleške namerno provereno.

Pre nastavka proveri svaku isporučenu datoteku:

```bash
find /private/tmp/lionjudo-video-review-session/session -type f -size 0 -print
find /private/tmp/lionjudo-video-review-session/session -type f \( -name '*.mp4' -o -name '*.mov' \) \
  -exec ffprobe -v error -show_entries format=duration,size -of compact=print {} \;
```

Pregledaj side-by-side i nezavisne plejere na početnom ankeru, drugom ankeru, jednom bacanju i preseku povrede. Oba prikaza moraju biti nenulta, iste globalne scene i bez vidljivog pomeranja. Ako nisu: vrati se na preview, ponovo izaberi oba trostruka tap-a, proveri smer mapiranja `iPhone -> Sony`, pa napravi novu praznu sesiju. Ne popravljaj neusaglašenost ručnim pomeranjem samo jednog klipa.

## Pokretanje lokalnog servera

Server sluša samo na loopback adresi. Port `0` bira prvi slobodan port:

```bash
python - <<'PY'
from pathlib import Path
from coach_app.server import create_server

server = create_server(Path("/private/tmp/lionjudo-video-review-session/session"), port=0)
print(server.base_url)
try:
    server.start_in_thread().join()
except KeyboardInterrupt:
    server.shutdown()
PY
```

U pregledaču proveri srpsko-latinične oznake, oba jednaka video prikaza, izbor događaja, sinhronizovano traženje i klik na grafikon, `Sačuvaj`, kao i neposredne `/izvestaj.csv` i `/izvestaj.md` izvoze. Povredni događaj mora imati oznaku `Prijavljen povredni događaj` i ostati samo za čitanje. To je namerno: snimak se čuva kao trag, ali se događaj ne koristi za normalnu statistiku niti za trening.

## Značenje metrika i ograničenja

Sve video metrike su **normalizovani opisi kretanja u ravni kamere**. One nisu fizički instrumenti.

| Metrika | Značenje | Granica tumačenja |
|---|---|---|
| `Brzina ulaska norm` | Promena položaja izabranih tačaka tela, skalirana veličinom tela i kadrom | Ne predstavlja brzinu u metrima u sekundi |
| `Rotacija trupa 2D (step/s)` | Promena 2D ugla linije ramena/kukova | Ne meri 3D ugaonu brzinu ili silu |
| `Promena visine kukova norm` | Relativna promena visine kukova u kadru | Perspektiva, zaklon i izbor poze mogu je promeniti |
| `Vreme oporavka (s)` | Vreme do povratka detektovanog kretanja ispod praga | Nije medicinsko vreme oporavka |
| `Intenzitet pokreta (0-100)` | Interno skalirana video energija kretanja | Nije sila, snaga, energija, vat, ubrzanje ili težina udarca |

Ne izvoditi zaključke o sili, snazi, vatima, fizičkom ubrzanju, težini udara ili medicinskoj dijagnozi. Oznaka tehnike je predlog/beleška trenera, ne automatska stručna presuda.

## Izvoz, privatnost i bezbednost izvora

- Čuvaj sesiju, preview i QA snimke u posebnom lokalnom direktorijumu kao što je `/private/tmp/lionjudo-video-review-session`.
- `review.json` beleži putanje i SHA-256 izvore radi provere porekla; hash nije kopija videa.
- Izvorne snimke sa memorijske kartice i iz `Downloads` samo čitaj. Ne premeštaj, ne preimenuj i ne briši ih.
- Ne šalji snimke, seed podatke, transkript ili izveštaje na mrežni servis. Server treba da bude `127.0.0.1`.
- Ne objavljuj raw ili `--no-blur` izlaz. Deca i sporedni sportisti ostaju zaštićeni pravilima privatnosti aplikacije.

## Automatizovana provera

Standardni `unittest` je merodavan:

```bash
python -m unittest tests.test_video_review_e2e -v
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q pipeline coach_app tests
node --check coach_app/static/app.js
git diff --check
```

Test koristi `tempfile` i mock-uje samo media subprocess granice, YOLO inferencu i Whisper poziv. Vremenska mapa, ugovor, isključenje povrede, importer, HTTP zahtevi, anotacija, CSV i Markdown ostaju stvarni.
