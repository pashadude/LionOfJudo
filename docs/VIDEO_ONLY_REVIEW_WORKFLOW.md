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

Tačne preflight komande za potvrđene kandidate, bez izmene izvora:

```bash
PREFLIGHT=/private/tmp/lionjudo-video-review-session/preflight

# Prvi trostruki tap: Sony 9.788125/10.192875/10.665250, iPhone 28.395750/28.685000/29.158500.
ffmpeg -hide_banner -loglevel error -ss 8.5 -i "$SONY" -t 3.5 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/sony_first_triple.mp4"
ffmpeg -hide_banner -loglevel error -ss 27.2 -i "$IPHONE" -t 3.5 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/iphone_first_triple.mp4"

# Drugi trostruki tap: Sony 110.862625/111.124000/111.426625, iPhone 130.359125/130.567875/130.878000.
ffmpeg -hide_banner -loglevel error -ss 109.5 -i "$SONY" -t 3.5 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/sony_second_triple.mp4"
ffmpeg -hide_banner -loglevel error -ss 129.1 -i "$IPHONE" -t 3.5 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/iphone_second_triple.mp4"

# Povreda: početak bacanja iPhone ~143.8 s, udar ~145.2 s, klečanje ~146.7 s; Sony cutoff 126.0 s.
ffmpeg -hide_banner -loglevel error -ss 124.5 -i "$SONY" -t 3.5 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/sony_injury_cutoff.mp4"
ffmpeg -hide_banner -loglevel error -ss 142.5 -i "$IPHONE" -t 5.0 \
  -vf "scale=960:-2" -c:v libx264 -crf 28 -an \
  "$PREFLIGHT/iphone_injury.mp4"

# Kontakt-tablice za brzu proveru sadržaja svakog kandidata i povrede.
for clip in "$PREFLIGHT"/*triple.mp4 "$PREFLIGHT"/*injury*.mp4; do
  name="${clip%.mp4}"
  ffmpeg -hide_banner -loglevel error -i "$clip" -vf "fps=2,scale=480:-2,tile=4x3" -frames:v 1 \
    -q:v 3 "${name}_contact.jpg"
done

find "$PREFLIGHT" -type f \( -name '*.mp4' -o -name '*.jpg' \) -print0 | \
  xargs -0 -n1 ffprobe -v error -show_entries format=duration,size -of compact=print
```

## Import i potvrde

Import zahteva dve potvrđene `AnchorPair` vrednosti. `sony_s` i `iphone_s` su sekunde na odgovarajućim fajlovima, `user_confirmed` mora biti `true`, a `triple_tap_count` tačno `3`. Primer poziva iz korena projekta:

```bash
cd /private/tmp/LionOfJudo-video-review
ANCHORS=/private/tmp/lionjudo-video-review-session/anchors.json
mkdir -p "$(dirname "$ANCHORS")"
printf '%s\n' '{"anchors":[{"name":"pocetak","sony_s":10.192875,"iphone_s":28.685000,"user_confirmed":true,"triple_tap_count":3},{"name":"kontrola","sony_s":111.124000,"iphone_s":130.567875,"user_confirmed":true,"triple_tap_count":3}]}' > "$ANCHORS"

/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py import \
  --sony /Volumes/Untitled/PRIVATE/M4ROOT/CLIP/C0007.MP4 \
  --iphone /Users/pauldudko/Downloads/IMG_3852.mov \
  --session-dir /private/tmp/lionjudo-video-review-session/session \
  --anchors-json "$ANCHORS" \
  --injury-cutoff-sony-s 126.000 \
  --blue-seed-sony 1897,887,2081,1486 \
  --analysis-fps 3.0 \
  --model-path /Users/pauldudko/VSProjects/LionOfJudo/yolo11x-pose.pt \
  --device mps \
  --event-threshold 0.4
```

Prvi potvrđeni anker je `10.192875` s na Sony osi, pa normalna pose analiza počinje tamo; pre-ankerski video se ne uključuje. Seed je potvrđen na Sony kadru `10.210` s i importeru se prosleđuju uglovi `(1897,887,2081,1486)` (oblik `x,y,w,h` je `1897,887,184,599`). `--analysis-fps 3.0` obrađuje deterministički svaki izabrani izvorni kadar po strajdu, čuva njegov izvorni timestamp i koristi efektivnu stopu za konačne razlike. To je grubo uzorkovanje video-pokreta; preskočeni kadrovi se ne izmišljaju. U ovom realnom prototipu `--event-threshold 0.4` je eksplicitno snižen da bi se uzorci na gruboj stopi obuhvatili; podrazumevana vrednost bez ove opcije ostaje `0.5`. Lokalna težina mora postojati; importer odbija nepostojeću težinu i ne preuzima model. Import pravi `review.json`, analize, preview fajlove, klipove događaja i `media/session_side_by_side.mp4`.

Uvezena sesija sa izvedenim klipovima, metrikama ili događajima ima `sync_locked: true`. Ankeri, afina mapa i presek povrede tada se ne menjaju u postojećoj sesiji; HTTP zahtev vraća `409` i traži novi uvoz. Minimalna sesija pre uvoza može menjati sinhronizaciju samo dok nema zavisne izvedene artefakte. Za bezbednu nadogradnju stare sesije bez ponovne YOLO obrade koristi `tools/video_review.py migrate --session-dir <sesija>`; migracija računa kanonske nizove iz postojećih `frame_metrics` i ne dodiruje izvore.

Ne pokreći import preko već anotirane sesije; za novu obradu koristi prazan direktorijum. `force_reimport=True` je dozvoljen samo kada je očuvanje postojeće trenerove beleške namerno provereno. Beleške se prenose samo na novi događaj sa istim `event_id`; neuparene beleške idu u `orphaned_annotations` sa izvornim ID-em i ne vraćaju stari događaj ili stari medijski put u aktivnu sesiju.

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
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py serve \
  --session-dir /private/tmp/lionjudo-video-review-session/session --port 0
```

U pregledaču proveri srpsko-latinične oznake, oba jednaka video prikaza, izbor događaja, sinhronizovano traženje i klik na grafikon, `Sačuvaj`, kao i neposredne `/izvestaj.csv` i `/izvestaj.md` izvoze. `Novi događaj`, `Primeni granice`, `Podeli`, `Spoji sa sledećim` i `Obriši` rade na Sony master osi; iPhone granice se računaju inverznom afinom mapom. Svaka uspešna izmena ponovo pravi oba klipa iz nepromenljivih izvora, proverava veličinu i trajanje, seče metrike iz sačuvanih kadrova, obnavlja izveštaje i tek zatim atomski menja `review.json`.

Promena granica čuva postojeću anotaciju. Podela je konzervativna: leva polovina zadržava anotaciju, desna počinje bez trenerove anotacije i bez glasovnog predloga. Spajanje susednih normalnih događaja čuva jedinu popunjenu ili dve identične anotacije; dve različite popunjene anotacije daju `409` dok ih trener ne uskladi. Brisanje anotiranog događaja premešta njegova polja u `orphaned_annotations`. Povredni događaj se ne može menjati, deliti, spajati ili obrisati i mora ostati samo za čitanje. To je namerno: snimak se čuva kao trag, ali se događaj ne koristi za normalnu statistiku niti za trening.

## Značenje metrika i ograničenja

Sve video metrike su **normalizovani opisi kretanja u ravni kamere**. One nisu fizički instrumenti.

Kanonski niz svakog `frame_metrics` zapisa koristi Sony `timestamp_s` i ključeve `brzina_ulaska_norm`, `rotacija_trupa_2d_dps`, `promena_visine_kukova_norm`, `sirina_stava_norm` i `intenzitet_pokreta_0_100`. Isti nazivi se koriste u Python serijalizaciji, `review.json`, izveštajima i JavaScript grafikonima. Intenzitet je stvarno izračunat uzorak u opsegu `0..100`, poravnat sa tim timestampom; nedostajući uzorak ostaje `null`, ne zamenjuje se izmišljenom vrednošću.

| Metrika | Značenje | Granica tumačenja |
|---|---|---|
| `Brzina ulaska norm` | Promena položaja izabranih tačaka tela, skalirana veličinom tela i kadrom | Ne predstavlja brzinu u metrima u sekundi |
| `Rotacija trupa 2D (step/s)` | Promena 2D ugla linije ramena/kukova | Ne meri 3D ugaonu brzinu ili silu |
| `Promena visine kukova norm` | Relativna promena visine kukova u kadru | Perspektiva, zaklon i izbor poze mogu je promeniti |
| `Širina stava norm` | Normalizovano 2D rastojanje između članaka u ravni kamere | Nije mera stabilnosti niti 3D razmak |
| `Vreme oporavka (s)` | Vreme do povratka detektovanog kretanja ispod praga | Nije medicinsko vreme oporavka |
| `Intenzitet pokreta (0-100)` | Interno skalirana video energija kretanja | Nije sila, snaga, energija, vat, ubrzanje ili težina udarca |

`Vreme oporavka (s)` se meri od timestamp-a najveće konačne energije pokreta u događaju do timestamp-a trećeg uzastopnog uzorkovanog opažanja sa energijom `<= 0.20`. Tačno tri susedna uzorka su potrebna; nedostajući uzorak, rupa u indeksima ili vrednost `> 0.20` prekida niz. Rezultat je `None` kada vrh ili takav niz nisu opaženi unutar događaja. Pri uzorkovanju od približno `3 FPS` to je transparentan video-deskriptor, ne medicinska procena oporavka.

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
