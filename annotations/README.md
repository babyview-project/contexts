# Activity annotations website

A web app where coders label short headcam clips with:

1. the primary activity the child is **doing** (`?type=do`) or **seeing** (`?type=see`), and
2. whether anyone is interacting with the child.

Each coder goes through **examples → practice → main clips**:

- **Examples** are clips with the correct answers and the reasoning behind them.
- **Practice** clips are graded against gold standards. Coders who don't pass get a new set, up to the maximum number of attempts.
- **Main clips** are assigned to each coder from the shared pool.

Training results are stored per coder ID and task, so a coder who has passed skips training next time.

## Quick start

```bash
cd annotations
npm install

npm run test-site            # test mode + dev mode: no videos or MongoDB needed
npm run test-site:prod-flow  # test mode with the real training gates (no skip buttons)
npm run test-site:accounts   # test mode with username/password login
npm run dev                  # real videos + MongoDB, with dev skip controls
npm start                    # production

npm run users -- add <username> [--admin]   # create a coder account (prints a generated password)
```

Then open:

| Page | URL |
|---|---|
| Annotation (choose task) | http://localhost:3000/experiment/activities.html |
| Doing task | http://localhost:3000/experiment/activities.html?type=do |
| Seeing task | http://localhost:3000/experiment/activities.html?type=see |
| Admin dashboard | http://localhost:3000/experiment/admin.html |
| Gallery (admin) | http://localhost:3000/experiment/gallery.html |

Send coders the `?type=do` or `?type=see` link. Without `type`, the page asks which task to do.

## Running on node14 (current deployment)

The site runs on `node14-ccn2cluster.stanford.edu` from `~/contexts/annotations`, this folder in the [contexts](https://github.com/babyview-project/contexts) repo checkout. `~/activity-annotations` is a symlink to it. Deployment state (`.env`, `users.json`, `db/`, `env/`, `clips/`, `data/`, `logs/`, `backups/`) lives in the same folder and is gitignored.

- **Address:** `http://node14-ccn2cluster.stanford.edu:3080/experiment/activities.html?type=do` (or `?type=see`). Admin is at `/experiment/admin.html`.
- **Who can reach it:** node14 has a private address, so only people on Stanford wifi or the Stanford VPN.
- **Node.js and MongoDB** come from a conda env in `env/`, installed from `/data/tsepuri/miniconda3`.
- **MongoDB** only listens on `127.0.0.1:27117`, with its data in `db/`.
- **`.env`** holds `PORT=3080`, the database address and `SESSION_SECRET`. It is readable by its owner only.

```bash
ssh node14-ccn2cluster.stanford.edu
cd ~/activity-annotations
scripts/server_ctl.sh status            # start | stop | restart | status | logs
PATH=$PWD/env/bin:$PATH npm run users -- add <username> [--admin]
```

The site and database run in the tmux sessions `annot-app` and `annot-db`, so they keep running after you log out. Logs are in `logs/`.

**Updating the code:** commit and push from `~/contexts` on node14, or `git pull` there after pushing from your laptop, then run `scripts/server_ctl.sh restart`. `gold_standards.csv` is tracked, and edits made in the gold standards editor or the gallery change it in place on node14. Commit them there, or a pull may conflict with them.

**Moving to a public server** (e.g. stanford-cogsci.org):
1. Copy the code and the `clips/` folder over.
2. Install Node.js and MongoDB, and set `.env`.
3. Copy `users.json` if the accounts should carry over.
4. To keep the existing annotations, also move the MongoDB data with `mongodump` / `mongorestore`.

## Modes

Modes are set with command-line flags (`node server.js --test --dev`) or environment variables (`TEST_MODE=true`, `DEV_MODE=true`).

| | **Test mode** (`--test`) | **Dev mode** (`--dev`) |
|---|---|---|
| Videos | Not needed. Missing clips show a placeholder with the filename. If there are no main videos, fake clips (`test_clip_001.mp4`, …) are used. | Real videos |
| Storage | In memory; everything is lost on restart | MongoDB |
| Login | Coders type an ID (no accounts), unless `AUTH_MODE=accounts` is set | As configured |
| Skip controls | Only with `--dev` | **Skip to: Examples / Practice / Main videos** in the header |
| Training gates | Enforced unless `--dev` | Relaxed: main clips are reachable without passing, and practice attempts are unlimited |
| What gets saved | Everything, in memory | Nothing from dev sessions: practice isn't recorded and main clips are an unsaved preview set (see below) |

**Dev mode from the URL:** add `&dev=1` (e.g. `activities.html?type=do&dev=1`) to turn on dev mode for one browser, without restarting the server with `--dev`.
- With accounts, this only works for **admin** accounts. Coders who add it see a notice and nothing changes; the server still requires them to pass training.
- Without accounts (`auth.mode: "none"`), it works for anyone.
- The admin dashboard has "Open in dev mode" links.

**Dev mode is a sandbox: nothing it does counts.** This applies to `--dev` and `?dev=1` alike.
- **Practice:** attempts are graded and show feedback, but aren't recorded. They don't use up attempts or count as passing training.
- **Main task:** you get a preview set of `videosPerCoder` clips that is never stored, so it uses no coder slots and doesn't affect coverage. Answers are checked but not saved; progress lasts until the page is reloaded.
- **🔀 Shuffle preview clips** in the main-task panel draws a new preview set. A reload keeps the current set for that browser tab.
- To code for real, an admin opens the page without `?dev=1`, which uses their real training status and assignment.

In test or dev mode, the task badge in the header has a **switch** link that restarts the page with the other task (doing ↔ seeing).

## Accounts

When `auth.mode` in `config.json` is `"accounts"` (the default), everyone logs in with a username and password from `annotations/users.json`.

- The username is the coder ID, so coders can't type in someone else's ID.
- Their progress follows their account, and they can log out from the header.
- Video files are only served to logged-in users.
- Only accounts marked as admin can open the admin dashboard.

```bash
npm run users -- add jdoe                 # new coder; prints a random password once
npm run users -- add alice --admin        # admin (can open admin.html)
npm run users -- add jdoe                 # run again to reset a password
npm run users -- add jdoe --password '…'  # choose the password yourself (min 8 characters)
npm run users -- remove jdoe              # annotations are kept
npm run users -- list
```

Changes apply right away; the server rereads `users.json` on each login.

**Passwords are never stored.** `users.json` holds only a salted scrypt hash of each password, so a password can't be read back out of the file. If someone forgets theirs, reset it with `add`.

- `users.json` is gitignored and is written readable by its owner only.
- Keep it that way: weak passwords could be guessed offline from a leaked file. The generated passwords are random enough that this isn't a concern.
- Logins are rate-limited: 10 failures for a username, or 50 from one IP address, locks it for 15 minutes.

Set `auth.mode` to `"none"` to have coders type their ID instead, with no login. Then anyone with the link can annotate and open the admin page, so use it only for testing.

Prolific or Sona would pass a participant ID in the URL instead. That would be another mode added next to these two.

## Clips

All clips live in one flat folder, `annotations/clips/`, which is gitignored. It holds the 200 main clips, the training (gold/example) clips, and their manifests, so the folder can be copied to another server as-is.

It is built on the cluster from the 2026.1 10s clips by `scripts/build_clip_folder.py`:

```bash
python3 scripts/build_clip_folder.py \
  --source /ccn2b/dataset/babyview/2026.1/10s_clips/video \
  --recordings data/recordings_processed.csv \
  --gold gold_standards.csv \
  --out clips --n-main 200 --seed 2026
```

- **Training clips.** `gold_standards.csv` originally used the old clip names, e.g. `00680001_2023-06-03_3_9d5449d617_processed_012.mp4`. These map to the new ones, e.g. `S00680001_2023-06-03_3_recL5nH9ztZzGJgKs_chunk0012of0027.mp4`. The video name maps through `superseded_gcp_name_feb25` → `unique_video_id` in `recordings_processed.csv`. Clip numbers are 0-based in both schemes, so `processed_012` is `chunk0012`.
- **Main clips.** 200 clips are drawn uniformly at random with a fixed seed, leaving out any recording that a training clip comes from. Rerunning with the same seed gives the same clips.
- **Outputs.** The script writes `clips/main_clips.csv`, `clips/training_clips.csv` (old → new name for each training clip) and `clips/gold_standards.csv` (new names, old ones in `old_video_filename`). Copy that gold CSV over `annotations/gold_standards.csv`.

**Moving to another server:** `rsync -a clips/ server:path/to/annotations/clips/`. No config changes are needed.

## Configuration

### `config.json`

```jsonc
{
  "auth": { "mode": "accounts" },  // "accounts" (users.json logins) or "none" (coders type an ID)
  "assignment": {
    "videosPerCoder": 50,  // clips per coder per task; null = every coder gets every clip
    "ratingsPerClip": 5    // a clip is never assigned to more than this many coders (per task)
  },
  "training": {
    "maxAttempts": 3,      // practice rounds before a coder is locked out
    "videosPerAttempt": 5, // gold standard clips per round
    "passThreshold": 0.75  // fraction correct to pass (a clip is correct when BOTH answers match)
  },
  "videos": {
    "mainDir": "clips",                       // served at /videos
    "mainListFile": "clips/main_clips.csv",   // .csv (video_filename column) or .txt (one per line); null = every video in mainDir
    "trainingDirs": ["clips"]                 // served at /training-videos
  },
  "test": { "fakeVideoCount": 12 }
}
```

**Main clip pool.** If `mainListFile` is set, the pool is the clips listed in it. Otherwise it is every video file in `mainDir`. Doing and seeing share the same pool, but assignments and rating counts are tracked separately for each task.

**Assignment.** The first time a coder reaches the main task, they are assigned `videosPerCoder` clips, picking the clips with the fewest coders so far. No clip goes to more than `ratingsPerClip` coders. Assignments are fixed once made: changing `videosPerCoder` later only affects new coders.

A clip counts as taken as soon as it is assigned, even if the coder never finishes it.

Planning for production (200 clips × 5 ratings = 1,000 ratings per task):

| `videosPerCoder` | Coders needed per task |
|---|---|
| 200 | 5 |
| 100 | 10 |
| 50 | 20 |
| 25 | 40 |

When every clip has `ratingsPerClip` coders, new coders are told there are no clips left.

### `activities.json`

This holds:

- the dropdown options for each task, in display order (`do`, `see`)
- a one-line description of each task
- a definition for each activity, separately for each task (`definitions.do`, `definitions.see`), plus one for "anyone interacting". The seeing versions describe what the child is watching, e.g. "walking" there means people in view are walking.

The definitions appear in the **ⓘ Definitions** button in the header of every page, under the dropdown when an activity is selected, and next to wrong answers on the practice feedback page.

The definitions are a first draft; please review them before launch. The server warns at startup if an activity has no definition.

### `gold_standards.csv`

| Column | Meaning |
|---|---|
| `video_filename` | Clip file (looked up in `trainingDirs`) |
| `order` | Display order for examples (blank = after the numbered ones) |
| `primary_activity` | Must be one of the task's activities in `activities.json` |
| `anyone_interacting` | `yes` / `no` |
| `type` | `example` (shown with answers), `gold` (practice pool); anything else is ignored, e.g. `candidate` (shortlisted from the gallery) |
| `negative_of` | Optional: the activity this clip is a negative example of (set from the gallery or the editor) |
| `disabled` | Optional: `yes` turns a `gold` / `example` row off without changing its type (set with **Stop using** / **Use again** in the editor or gallery) |
| `modality` | `do` / `see` |
| `pa_reasoning`, `ai_reasoning` | Reasoning shown on example clips and practice feedback |

Rows whose label is not in the activity list, or whose video file is missing (outside test mode), are **skipped**. Skipped rows are listed in the server log and at the top of the admin dashboard.

Each coder's practice clips come from a fixed shuffle of the gold pool, so reloading gives the same set. A set never has two clips with the same primary activity. Each attempt takes clips the coder hasn't seen yet first, and only reuses earlier ones when the remaining clips don't cover enough different activities.

**Gold standards editor:** `/experiment/admin.html` → **Gold standards** (admins only).
- It shows each row's clip next to its label, interaction answer and reasoning. By default it lists only the rows the app is skipping, such as labels that aren't in the activity lists.
- **Save** rewrites only that row of `gold_standards.csv` on the server and keeps the previous file in `backups/`. Training picks up the change immediately.
- Each row says whether the app uses it. **Stop using** turns a gold or example row off (`disabled = yes`) and **Use again** turns it back on, keeping its type. Rows of other types, like `candidate`, have **Use as gold** / **Use as example**. These save right away. Main-task clips can't be made gold or example.
- The server copy is now the newest. Use **Download gold_standards.csv** to copy it back into the repo before syncing code to the server again, or the sync will overwrite your edits.

If you edit the CSV by hand instead, restart the server or call `POST /api/training/reload-gold-standards`.

### Environment (`annotations/.env`)

| Variable | Default | |
|---|---|---|
| `PORT` | `3000` | |
| `MONGODB_URI` | `mongodb://localhost:27017/activity_annotations` | |
| `SESSION_SECRET` | `secret-key-temp` | **Set this to a long random string in production**; it signs login cookies |
| `AUTH_MODE` | from `config.json` | Override `auth.mode` (`accounts` / `none`) |
| `USERS_PATH` | `users.json` | Alternate accounts file |
| `CREDENTIALS_PATH` | `credentials/` | Folder with `ssl_key.pem` + `ssl_cert.pem`; HTTP is used if they're missing |
| `CLIENT_URL` | | Only needed if the page is served from a different origin |
| `CONFIG_PATH`, `ACTIVITIES_PATH` | `config.json`, `activities.json` | Use alternate config files |

## Checking who has completed training

Open **`/experiment/admin.html`** and log in with an admin account. For every coder ID it shows, per task:

- training status (not started / in progress / passed / failed) and number of attempts
- best practice score
- main clips completed out of clips assigned

It also shows coverage per task: ratings done against the target, and how many clips have 0, 1, 2, … ratings.

The page can export all main or all training annotations as CSV.

The same data is available as JSON:

```bash
curl -s -c /tmp/admin.jar -H 'Content-Type: application/json' \
     -d '{"username":"alice","password":"…"}' localhost:3000/api/auth/login
curl -s -b /tmp/admin.jar localhost:3000/api/admin/status \
  | jq '.coders[] | {name, do: .do.training.passed, see: .see.training.passed}'
```


## Gallery: finding new gold standards and negative examples

`/experiment/gallery.html` (admins only; linked from the admin dashboard) shows clips as a grid of videos with every answer for each clip. Use it to find new gold standard clips, especially **negative examples**: clips that look like an activity but aren't it.

There are two sources:

- **Main clips (this app):** the main-task pool, including clips nobody has annotated yet, with the answers saved here. Filters:
  - activity + "someone said it, the majority didn't" (candidate negatives), "anyone said it", "the majority said it", or "the model detected it, the majority didn't"
  - agreement (not unanimous, or 50% or less), interaction disagreement, and the number of answers
- **All model detections:** all ~300k clips the model labeled (`all_contexts_combined.csv`). Clips checked in an earlier pass show those answers, from the **model validation** pass (DD/JY/VM, with description accuracy 1–5) and the **original doing/seeing pass** (elenahc/m1deng, …, with confidence 1–3). Filters:
  - "the model detected it, earlier coders mostly didn't" (candidate negatives)
  - "earlier coders saw it, the model labeled something else"
  - which pass checked the clip (or never checked)
  - the model's raw label
  - sort by **least validated first**: the mean of scaled description accuracy and the share of earlier answers that include the model's activity

Each card shows the model's description of the clip, when the clip is in the model's set; click it to show all of it. Clips already in `gold_standards.csv` have a ★ chip; on gold and example rows, **stop using** / **use again** turns the row off or on for training.

Older label sets are matched against the current lists through `lib/labels.js`. For example, "playing" counts toward every "playing with…" activity and "meal time" counts as "eating/drinking".

**Adding a clip to the gold standards:** use **＋ Add to gold standards** on a card.
- The form fills in the majority answer, and fills in **Negative example of** with the activity you filtered on. That is saved in a `negative_of` column.
- Clips from the model detections are first copied into `clips/` under their 2026.1 name, with the old name in `old_video_filename`.
- Add clips as **candidate** to shortlist them. The app ignores that type. Change the type to `gold` or `example` later in the gold standards editor, which now also edits `type` and `negative_of`.
- Main-task clips can only be added as `candidate` or `confusing`. As practice clips, coders would see the answer before annotating them.
- Adding a `gold` clip reshuffles the practice sets of coders who are still in training.

**Building the gallery data:** run `python3 scripts/build_gallery_data.py` on node14. It reads `~/contexts/data/completed_annotations/` and `~/contexts/data/all_contexts_combined.csv`, maps clips to `/ccn2b/dataset/babyview/2026.1/10s_clips/video`, and writes `data/original_pass.json` and `data/model_detections.tsv` (~110 MB, gitignored). It takes about 10 seconds, and the server picks up a new build without a restart. The model-detection clips are streamed from `/ccn2b`, so they only play on the cluster.

## Data (MongoDB collections)

| Collection | Contents |
|---|---|
| `annotations` | One document per coder × clip × task. Training responses have `isTraining: true` and an `attemptNumber`. |
| `trainingattempts` | One per practice round: score, passed, per-clip correctness |
| `assignments` | The clips assigned to each coder for each task, in presentation order |
| `users` | Coder IDs and when they were first seen |

## Files

```
server.js              Express server and API routes
lib/config.js          Loads config.json / activities.json; --test / --dev flags
lib/store.js           MongoDB store, plus the in-memory store used in test mode
lib/gold.js            Gold standard loading, practice-set selection, grading
lib/users.js           Password hashing, users.json access, login rate limiting
lib/labels.js          Maps older activity labels (original pass, model) onto the current lists
scripts/users.js       `npm run users` account management
users.json             Coder accounts (gitignored; created by `npm run users -- add`)
scripts/build_clip_folder.py   Builds clips/ from the cluster's 10s clips (see Clips)
scripts/build_gallery_data.py  Builds data/ for the gallery's earlier passes and model detections
clips/                 Main + training clips and manifests (gitignored)
public/activities.*    Annotation page
public/admin.*         Admin dashboard
public/gallery.*       Annotation gallery (admin)
public/gold.*          Gold standards editor (admin)
```
