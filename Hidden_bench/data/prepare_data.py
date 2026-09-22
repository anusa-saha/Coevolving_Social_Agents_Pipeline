"""prepare_data.py - copy the 1100D split into this folder and stamp a globally unique uid.

The 1100D files in ICLR/1100D/ carry no `uid`. env.load_scenarios falls back to the row
index when a file has none, which is unique WITHIN a file but would make train row 0 and
test row 0 both uid=0 - the 450D convention (uid = index in the pooled dataset) exists
precisely so ids stay unique and traceable ACROSS the two files. So this assigns:

    uid = index in the pooled, deterministically ordered 1100-scenario set

Train keeps no eval_group; test keeps the seen/unseen tag the split wrote.

    python prepare_data.py

Writes data/1100_train.json (720) and data/1100_test.json (380 = 180 seen + 200 unseen),
and hard-fails if train and test overlap by uid, by identity, or by content.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "1100D"))


def content_key(s):
    """Identity of a scenario independent of the bookkeeping fields we add/strip."""
    skip = {"uid", "eval_group"}
    body = {k: v for k, v in s.items() if k not in skip}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load(name):
    with open(os.path.join(SRC, name), encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    train = load("1100_train.json")
    test = load("1100_test_all.json")
    assert len(train) == 720, len(train)
    assert len(test) == 380, len(test)

    # uid = position in the pooled set, ordered deterministically so a re-run is stable
    # and independent of the shuffle inside either file.
    pool = sorted(train + test,
                  key=lambda s: (s.get("domain", ""), s.get("scenario_id", "")))
    uid_of = {}
    for i, s in enumerate(pool):
        k = content_key(s)
        if k in uid_of:
            raise SystemExit("[FAIL] duplicate scenario in the pooled 1100 set")
        uid_of[k] = i

    for s in train:
        s["uid"] = uid_of[content_key(s)]
        s.pop("eval_group", None)
    for s in test:
        s["uid"] = uid_of[content_key(s)]

    # ---- the overlap check, three independent ways ----
    tr_uid, te_uid = {s["uid"] for s in train}, {s["uid"] for s in test}
    tr_key = {content_key(s) for s in train}
    te_key = {content_key(s) for s in test}
    tr_sid = {(s.get("domain"), s.get("scenario_id")) for s in train}
    te_sid = {(s.get("domain"), s.get("scenario_id")) for s in test}

    assert len(tr_uid) == len(train), "duplicate uid inside train"
    assert len(te_uid) == len(test), "duplicate uid inside test"
    for what, a, b in (("uid", tr_uid, te_uid), ("content", tr_key, te_key),
                       ("domain+scenario_id", tr_sid, te_sid)):
        if a & b:
            raise SystemExit("[FAIL] train/test overlap by {}: {} shared".format(
                what, len(a & b)))
        print("  no overlap by {:<20s} ({} train vs {} test, 0 shared)".format(
            what, len(a), len(b)))

    trained = {s["domain"] for s in train}
    unseen = sorted({s["domain"] for s in test} - trained)
    n_seen = sum(1 for s in test if s["eval_group"] == "seen")
    n_unseen = len(test) - n_seen
    print("  train {} ({} domains) | test {} = {} seen + {} unseen".format(
        len(train), len(trained), len(test), n_seen, n_unseen))
    print("  held-out domains (never in train): {}".format(", ".join(unseen)))
    assert n_seen == 180 and n_unseen == 200
    assert all(s["domain"] in trained for s in test if s["eval_group"] == "seen")
    assert all(s["domain"] not in trained for s in test if s["eval_group"] == "unseen")

    for name, data in (("1100_train.json", train), ("1100_test.json", test)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("  wrote {} ({})".format(name, len(data)))


if __name__ == "__main__":
    main()
