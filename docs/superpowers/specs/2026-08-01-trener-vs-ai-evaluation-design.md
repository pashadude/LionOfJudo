# Trener vs AI Evaluation Design

## Cilj

Proširiti lokalni LionOfJudo pregled tako da trener prvo zabeleži nezavisnu
procenu bacanja, a zatim u režimu `Trener vs AI` vidi i osporava proverljiv,
deterministički predlog sistema. Sistem ostaje na srpskoj latinici, radi bez
LLM-a i čuva podatke potrebne za kasnije LoRA treniranje.

Aktivna probna sesija sadrži dva trenerom potvrđena bacanja, redom
`Tai-otoshi` i `Morote-seoi-nage`, nakon kojih sledi povredni događaj. Nijedno
bacanje nema unapred dodeljenu trenerovu ocenu.

## Principi

- Trenerova procena nastaje pre otkrivanja AI predloga, da AI ne usidri
  stručnu odluku.
- AI predlog je eksperimentalna procena izvedbe, ne autoritativna
  identifikacija ili ocena džudo tehnike.
- Svaka AI tvrdnja mora imati sledljiv dokaz: naziv metrike, vrednost,
  jedinicu i Sony sekundu.
- Bez dovoljnog kvaliteta praćenja sistem ne dodeljuje AI ocenu.
- Interfejs koristi naslov `IMU merenje (eksperimentalno)` i podnaslov
  `Prototip v1. Moguća velika greška.`
- Poreklo video-proxy vrednosti ostaje u strukturiranim internim i izvoznim
  metapodacima, ali se dodatno tehničko objašnjenje ne ponavlja u glavnom
  interfejsu.
- Originalni video-fajlovi se nikada ne menjaju. Aplikacija prikazuje samo
  izvedene medije sa zamagljenim licima.

## Tok Trener vs AI

### 1. Nezavisna procena trenera

Pre otkrivanja AI rezultata trener vidi sinhronizovane isečke, grafikone i
postojeće podatke o događaju. Unosi:

- potvrđen naziv tehnike;
- ocenu od 1 do 5;
- razlog svoje ocene.

Klikom na `Zaključaj procenu` nastaje nepromenljiva pre-AI revizija sa
vremenom zaključavanja. Naziv, ocena i razlog su obavezni za zaključavanje.
Razlog mora da sadrži najmanje jedan Sony trenutak. Dugme
`Dodaj trenutnu sekundu` unosi vreme trenutnog kadra u razlog, tako da zahtev
za preciznošću ne usporava rad trenera.
Ako bacanje nije dovoljno vidljivo, trener umesto ocene bira
`Nedovoljno vidljivo`; takav događaj ne učestvuje u poređenju ocena.

Prva zaključana procena zauvek ostaje revizija sa fazom `pre_ai`. Ispravka
posle otkrivanja pravi novu reviziju sa fazom `post_ai_korekcija` i ne menja
prvu. Tako se čuvaju nezavisna procena, naknadna korekcija i podaci za buduće
treniranje.

### 2. Otkrivanje mašinske procene

Posle zaključavanja interfejs otkriva blok `Trener vs AI` sa dve jasno
razdvojene strane:

- `TRENER`: zaključana tehnika, ocena i razlog;
- `AI`: predložena ocena, pouzdanost i razlog sa dokazima.

Interfejs prikazuje razliku ocena kao direktan izazov, ali ne proglašava
pobednika bez spoljne referentne ocene. Tekst poziva trenera da
`Potvrdi ili ospori AI`, bez tvrdnje da je viša ocena bolja procena.
Kada se ocene razlikuju, glavni poziv glasi
`AI odstupa za X poena. Odbrani procenu.`

AI strana počinje odeljkom `ČINJENICE SISTEMA`. Svaki red je numerički dokaz
sa Sony sekundom, a ne opšta tvrdnja. Trener zato vidi da sistem raspolaže
konkretnim podacima i mora da odgovori preciznim opažanjem iz istog snimka.

AI procena, eksperimentalni IMU pokazatelji i AI dokazni redovi postoje u
internom `review.json`, ali se pre zaključavanja rediguju iz odgovora
trenerskog API-ja, HTML podataka i CSV/Markdown izvoza. Otkrivanje atomski
upisuje `ai_otkriven_u` u AI zapis aktivne verzije događaja; tek zatim
trenerski API i izveštaji vraćaju AI polja za taj događaj.

### 3. Procena AI predloga

Trener bira tačno jednu vrednost:

- `Slažem se`;
- `Delimično`;
- `Ne slažem se`.

Može dodati opcioni razlog. Ova procena AI predloga je odvojena od originalne
trenerove ocene i razloga. Ako je AI odustao zbog nedovoljnih podataka,
kontrola odnosa ostaje dostupna, ali prikazuje i izbor `AI nema dovoljno
podataka` kao sistemski status, ne kao četvrtu trenerovu ocenu.

Svaki AI dokaz može da se označi kao `prihvatam`, `nepotpun` ili `osporavam`.
Ovo je opciono po pojedinačnom dokazu, ali daje strukturirane teške primere za
budući model i pretvara neslaganje u konkretan stručni odgovor.

## Deterministički AI evaluator

Evaluator ne koristi LLM. Za svaki događaj uzima samo kadrove iz njegovih
Sony granica i samo izabrani trag Dušana u plavom. Iz vremenske serije računa
dokaze na konkretnim Sony sekundama:

- vršnu ugaonu brzinu trupa u stepenima po sekundi;
- proxy linearnog ubrzanja od 0 do 100;
- proxy impulsa pokreta od 0 do 100;
- intenzitet kretanja od 0 do 100;
- dominantni smer rotacije;
- trenutak vršne aktivnosti;
- promenu visine kukova i širinu stava kada su dostupne.

### Kanonske 2D serije v1

Ulaz je izabrani trag sa 17 COCO tačaka. Indeksi ramena su `5,6`, kukova
`11,12`, a članaka `15,16`. Tačka je dostupna kada je njena pouzdanost najmanje
`0.30`. Središte para je aritmetička sredina njegovih koordinata. Dužina trupa
je euklidsko rastojanje između središta ramena i kukova u istom uzorku.

Za dva susedna konačna uzorka sa razlikom vremena `dt`:

- `brzina_ulaska_norm = distance(hip[i], hip[i-1]) / torso[i] / dt`;
- ugao ramena je
  `atan2(right_shoulder_y-left_shoulder_y,
  right_shoulder_x-left_shoulder_x)` u stepenima;
- `rotacija_trupa_2d_dps = wrap(angle[i]-angle[i-1]) / dt`, gde `wrap`
  vraća ugao u intervalu `[-180,180)`;
- promena visine kukova je
  `(hip_y[i]-first_valid_hip_y) / torso[i]`;
- širina stava je `distance(left_ankle,right_ankle) / torso[i]` kada su oba
  članka dostupna.

Unutrašnja praznina od najviše pet uzoraka linearno interpolira centre,
dužinu trupa, ugao i širinu stava. Veća ili rubna praznina ostaje `null`, a
izvod se ne računa preko nje. Nema drugog filtriranja osnovnih serija.
Po-uzorku se računa
`intensity = 0.50*clamp(speed/4*100) +
0.50*clamp(abs(rotation)/540*100)`, pa se intenzitet zamenjuje centriranom
aritmetičkom sredinom najviše tri susedna konačna uzorka. Ova definicija je
`video-pose-metrics-v1`; promena bilo kog pravila zahteva novi identitet.

Evaluator `deterministicki-v1` radi na nominalnih 6 uzoraka u sekundi, ali
svaki izvod računa iz stvarnih `timestamp_s` razlika. Validan uzorak ima
`vidljivo=true`, konačne centre ramena i kukova i pouzdanost obe tačke svakog
para najmanje `0.30`. Interpolirani uzorak može da učestvuje u vrednosti, ali
se ne računa kao validan za pokrivenost.

Za događaj se izračunavaju sledeće vrednosti, uvek iz konačnih uzoraka i sa
linearnom interpolacijom samo za unutrašnje praznine do pet uzoraka:

- `speed_peak`: 90. percentil `brzina_ulaska_norm`;
- `rotation_peak`: 90. percentil apsolutne `rotacija_trupa_2d_dps`;
- `acceleration_peak`: 90. percentil
  `abs(speed[i] - speed[i-1]) / (time[i] - time[i-1])`;
- `impulse_proxy`: trapezno integrisana apsolutna promena brzine kroz
  događaj;
- `intensity_peak`: 90. percentil kanonskog intenziteta v1.

Za sve percentile v1 koristi nearest-rank pravilo: sortirati konačne uzorke i
uzeti element `ceil(0.90*n)-1`; dokaz nasleđuje Sony vreme tog stvarnog
uzorka, a kod izjednačenja se bira ranije vreme. `impulse_proxy` je
`sum(abs(speed[i]-speed[i-1]))`, što je diskretni integral apsolutnog izvoda
brzine. Dominantni smer je znak zbira svih konačnih vrednosti rotacije, a
trenutak vršne aktivnosti je najranije vreme maksimalnog intenziteta.

Fiksne normalizacije v1 su:

- `speed_0_100 = clamp(speed_peak / 4.0 * 100)`;
- `rotation_0_100 = clamp(rotation_peak / 540.0 * 100)`;
- `acceleration_0_100 = clamp(acceleration_peak / 24.0 * 100)`;
- `impulse_0_100 = clamp(impulse_proxy / 12.0 * 100)`;
- `intensity_0_100 = clamp(intensity_peak)`.

`clamp` ograničava vrednost na `0..100`. Prototipski zbir je
`0.20*speed + 0.25*rotation + 0.20*acceleration + 0.15*impulse +
0.20*intensity`. Ocena je `1` za zbir manji od 20, `2` za `20..<40`, `3`
za `40..<60`, `4` za `60..<80` i `5` za `80..100`. Ovi pragovi nisu
tehnički standard džudoa; njihov identitet i sve konstante čuvaju se uz
rezultat, tako da promena pravila zahteva novi `evaluator_id`.
Svi numerički izlazi se za čuvanje zaokružuju na šest decimala, a Sony vremena
na tri decimale.

Evaluator prvo proverava kvalitet. Očekivani broj uzoraka je
`max(1, floor((sony_end_s-sony_start_s)*effective_analysis_fps)+1)`.
`coverage` je broj neinterpoliranih validnih uzoraka podeljen očekivanim
brojem uzoraka u granicama događaja.
`continuity` je `1 - clamp(najduža_nevalidna_praznina_s / trajanje_s)`.
Zbirna pouzdanost je `0.75*coverage + 0.25*continuity`. Rezultat je jedan od:

- `dostupno`: najmanje 12 validnih uzoraka, `coverage >= 0.70`, najduža
  nevalidna praznina najviše `0.50 s` i zbirna pouzdanost najmanje `0.70`;
  izlaz ima AI ocenu 1-5, razlog i najmanje dva dokaza;
- `niska_pouzdanost`: najmanje 6 validnih uzoraka i `coverage >= 0.35`, ali
  uslov za `dostupno` nije ispunjen; eksperimentalni IMU pokazatelji se
  prikazuju sa oznakom niske pouzdanosti, ali AI ocena ostaje prazna;
- `nedovoljno_podataka`: uslov za `niska_pouzdanost` nije ispunjen; nema
  ocene, a razlog navodi broj validnih uzoraka, pokrivenost i najdužu prazninu.

Razlog ne sme da tvrdi da je izmerena fizička sila ili snaga u vatima.
Vidljivo polje `3D pokazatelj snage` ostaje bez numeričke vrednosti i prikazuje
`Biće kalibrisano u sledećoj verziji.`

## Ugovor podataka

Svaki normalni događaj dobija sledeće logičke celine:

```json
{
  "event_revision": 1,
  "analysis_fingerprint": "sha256:...",
  "ai_procene": [
    {
      "event_revision": 1,
      "analysis_fingerprint": "sha256:...",
      "status": "dostupno | niska_pouzdanost | nedovoljno_podataka",
      "predlozena_ocena": 4,
      "pouzdanost_0_1": 0.82,
      "razlog": "...",
      "dokazi": [
        {
          "metrika": "ugaona_brzina_trupa_2d",
          "vrednost": 312.4,
          "jedinica": "step/s",
          "sony_s": 130.42
        }
      ],
      "evaluator_id": "deterministicki-v1",
      "ai_otkriven_u": "ISO-8601 | null"
    }
  ],
  "imu_eksperimentalno": {
    "ugaona_brzina_trupa_dps": 312.4,
    "proxy_ubrzanja_0_100": 71.0,
    "proxy_impulsa_0_100": 64.0,
    "intenzitet_0_100": 68.0,
    "dominantna_rotacija": "levo",
    "vrh_sony_s": 130.42,
    "pouzdanost": "visoka | srednja | niska",
    "izvor": "video_pose_proxy_v1",
    "snaga_3d": null,
    "snaga_3d_status": "Biće kalibrisano u sledećoj verziji."
  },
  "trener_procene": [
    {
      "revizija": 1,
      "faza": "pre_ai",
      "event_revision": 1,
      "analysis_fingerprint": "sha256:...",
      "status_vidljivosti": "dovoljno_vidljivo | nedovoljno_vidljivo",
      "potvrdena_tehnika": "Tai-otoshi",
      "ocena": 4,
      "razlog": "...",
      "citirani_sony_trenuci_s": [130.42],
      "zakljucano_u": "ISO-8601"
    }
  ],
  "aktivna_trener_revizija": 1,
  "procene_ai_predloga": [
    {
      "event_revision": 1,
      "analysis_fingerprint": "sha256:...",
      "trener_revizija": 1,
      "evaluator_id": "deterministicki-v1",
      "odnos": "slazem_se | delimicno | ne_slazem_se",
      "razlog": "...",
      "procene_dokaza": [
        {
          "metrika": "ugaona_brzina_trupa_2d",
          "odnos": "prihvatam | nepotpun | osporavam"
        }
      ],
      "sacuvano_u": "ISO-8601"
    }
  ],
  "aktivni_duel": {
    "event_revision": 1,
    "analysis_fingerprint": "sha256:...",
    "trener_revizija": 1,
    "evaluator_id": "deterministicki-v1"
  }
}
```

`predlozena_ocena`, numerički IMU proxy pokazatelji i trenerova ocena mogu
biti `null` u nezaključenom ili nepouzdanom stanju. Ako je
`status_vidljivosti=nedovoljno_vidljivo`, `ocena`, `razlog` i
`citirani_sony_trenuci_s` smeju biti prazni; takva zaključana revizija nije
etiketa kvaliteta izvedbe. API ne zamenjuje `null` podrazumevanom ocenom.
Svaka `procena_ai_predloga` je vezana za nepromenljivu trenersku reviziju,
verziju događaja, fingerprint analize i konkretnu verziju evaluatora.
Povredni događaj ostaje samo za čitanje i van statistike.

`analysis_fingerprint` je SHA-256 kanonskog JSON-a koji sadrži potpise oba
izvorna videa, Sony/iPhone granice, izabrani track ID, efektivni FPS,
`video-pose-metrics-v1` i `evaluator_id`. Promena granica ili izvora povećava
`event_revision`, pravi novi fingerprint i otvara novi krug procene. Stari
duel ostaje u nizovima istorije, a novi AI rezultat je ponovo skriven dok
trener ne zaključa novu `pre_ai` reviziju za novu verziju događaja.

## Interfejs

Stranica zadržava radni, gust raspored bez marketinškog hero odeljka.
Za izabrani događaj prikazuje:

1. dva sinhronizovana, zamagljena videa;
2. zajedničku vremensku kontrolu i korekciju granica;
3. grafikone kretanja;
4. nezavisnu procenu trenera ili njenu zaključanu reviziju;
5. posle zaključavanja, duel `Trener vs AI`;
6. `IMU merenje (eksperimentalno)` sa stabilnim numeričkim pločicama;
7. `ČINJENICE SISTEMA`, tabelu AI dokaza sa vrednostima i Sony sekundama;
8. kontrolu `Slažem se / Delimično / Ne slažem se` i opcioni razlog.

Segmentirana kontrola se koristi za odnos prema AI. Ocena 1-5 koristi pet
jasnih numeričkih izbora. Dugi razlozi se prelamaju i ne menjaju dimenzije
video-kontrola ili grafikona. Na mobilnom prikaz prelazi u jednu kolonu bez
horizontalnog prelivanja.

## Privatnost i izvedeni mediji

Sva lica zamagljuju se u:

- Sony i iPhone klipovima događaja;
- pregledima sinhronizacionih ankera;
- zajedničkom side-by-side videu;
- budućim klipovima izvoza za trenera.

Generisanje radi u režimu `blur all`, bez izuzetka za Dušana: YOLO pose
regioni glave i YuNet detektor obrađuju svaki kadar. Zatim verifikacioni
prolaz ponovo pokreće oba detektora sa pragom `0.30`; svaki preostali kandidat
se dodatno zamagljuje i verifikacija se ponavlja. Izlaz dobija
`privacy_verified=true` samo kada oba detektora mogu da obrade svaki kadar i
poslednji prolaz ne nađe nezamagljen kandidat. Za aktivnu sesiju se dodatno
vizuelno proverava kontaktni list sa početka, sredine i kraja svakog klipa iz
obe kamere.

Obrada je `fail closed`: ako zamagljena verzija ne može da se napravi ili
dobije `privacy_verified=true`, aplikacija ne koristi nezamagljeni izvedeni
klip kao zamenu.
Originalni Sony i iPhone fajlovi ostaju na postojećim lokacijama, ne menjaju
se i ne serviraju se direktno preko lokalnog HTTP servera.

## Izveštaji i budući LoRA model

CSV i Markdown izvoz uključuju:

- granice događaja i potvrđenu tehniku;
- AI status, ocenu, pouzdanost i razlog;
- svaki AI dokaz sa vremenom;
- eksperimentalne IMU pokazatelje i interno poreklo;
- zaključanu pre-AI trenerovu procenu;
- odnos trenera prema AI i opcioni razlog;
- trenerov odgovor na svaki pojedinačni AI dokaz;
- status vidljivosti i isključenje povrednog događaja.

Pre-AI procena, sve kasnije revizije, AI predlog i post-AI odgovor ostaju
odvojeni. Budući LoRA proces koristi samo reviziju sa `faza=pre_ai` kao
nezavisni cilj, a neslaganja i razloge kao skup teških primera, bez
prepisivanja istorijskih podataka.

## Migracija aktivne sesije

- Mapa vremena ostaje `Sony = iPhone - 3.0 s`.
- `Tai-otoshi`: Sony `128.5-132.0 s`, iPhone `131.5-135.0 s`.
- `Morote-seoi-nage`: Sony `132.8-135.0 s`, iPhone `135.8-138.0 s`.
- Povredni događaj: Sony `135.0-136.0 s`, iPhone `138.0-139.0 s`.
- Oba naziva imaju status `trener`, ali ocena i razlog ostaju prazni dok ih
  trener ne zaključa.
- Stara automatska zbirna metrika ne sme da se koristi za nove ručne granice.
  Svi sažeci se ponovo računaju iz vremenske serije; bez dovoljno uzoraka
  događaj dobija `niska_pouzdanost` ili `nedovoljno_podataka`.
- Aktivni događaji i izveštaji ne sadrže prethodno pogrešan `O-soto-gari`.

## Greške i oporavak

- Nevalidna ili nepotpuna trenerova procena ne može da se zaključa.
- Čuvanje procene, revizije i izveštaja je atomsko.
- Neuspela regeneracija klipa ne menja postojeći događaj ni njegov medij.
- Nedostupna metrika ne postaje nula; čuva se kao `null` i objašnjava u AI
  statusu.
- Promena granica događaja, traga subjekta ili verzije evaluatora pravi novi
  fingerprint i krug procene. Stari duel ostaje nepromenjen, a novi AI
  rezultat ne poredi se sa starom trenerovom revizijom.
- Povredni događaj se ne može oceniti, spojiti, podeliti ili obrisati.

## Provera prihvatljivosti

Implementacija je prihvatljiva kada su dokazani sledeći uslovi:

- prazna trenerova ocena ostaje `null`, nikada automatski `3` ili `0`;
- trener ne vidi AI procenu pre zaključavanja sopstvene procene;
- pre zaključavanja trenerski API i CSV/Markdown izvoz ne otkrivaju AI ocenu,
  razlog, IMU proxy vrednosti niti AI dokaze;
- prva zaključana revizija zauvek ostaje `faza=pre_ai`; svaka naknadna
  korekcija dobija novi broj i `faza=post_ai_korekcija`;
- promena granica posle zaključavanja povećava `event_revision`, menja
  fingerprint i zahteva novu nezavisnu trenersku procenu pre novog otkrivanja;
- trener ne može da zaključa procenu bez razloga i najmanje jedne Sony
  sekunde, a trenutna sekunda može da se doda jednim klikom;
- otkrivanje prikazuje dve strane i razliku ocena bez lažnog pobednika;
- neslaganje prikazuje poziv `AI odstupa za X poena. Odbrani procenu.` i
  numeričke `ČINJENICE SISTEMA`;
- svaki dostupan AI razlog citira najmanje dve konačne vrednosti i Sony
  sekunde unutar granica događaja;
- nepouzdano praćenje ne proizvodi AI ocenu;
- isti ulaz i `evaluator_id=deterministicki-v1` daju tačno isti sačuvani JSON
  proxy vrednosti, pouzdanosti, dokaza i ocene posle propisanog zaokruživanja;
- eksperimentalni IMU blok prikazuje traženi naslov, podnaslov, proxy vrednosti
  i status buduće 3D kalibracije;
- svi trenerovi odgovori opstaju posle ponovnog učitavanja i ulaze u oba
  izveštaja;
- svaki post-AI odgovor čuva vreme, event revision, analysis fingerprint,
  trenersku reviziju i evaluator ID koje je trener zaista video;
- svi servirani video-klipovi su zamagljene izvedene kopije, dok se originalni
  potpisi i veličine ne menjaju;
- aktivna sesija prikazuje tačno dva bacanja pravilnim redosledom i zatim
  povredni događaj;
- pre prve trenerske akcije `Tai-otoshi` i `Morote-seoi-nage` imaju
  trenerovu ocenu `null`, bez podrazumevane trojke;
- testovi API-ja, evaluatora, izveštaja, medija i pregledača prolaze;
- desktop i mobilni snimci ekrana nemaju preklapanje, prelivanje ili prazne
  video/grafičke površine.
