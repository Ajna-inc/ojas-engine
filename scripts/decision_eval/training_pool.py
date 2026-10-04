#!/usr/bin/env python3
"""Write a pool of training requests for `decision_rl`, with no dataset behind it.

Two kinds of file go into `<out>/`, one per family, in the trainer's JSONL form
(`{"body": request, "gold": {question id: key}}`):

* the benchmark's own task families, regenerated with other seeds so the items are
  not the benchmark's (any item equal to a benchmark item is dropped) and the answers
  are known, as the benchmark knows them;
* open states — support messages, reviews, records, incidents — assembled from
  templates, with generic questions and no answers: the teacher judges them.

Families whose content is finite (the capitals, the animal classes) regenerate the
same items in another order, so they contribute nothing here; a model learns them
from the open states and the teacher instead.

    training_pool.py <out> [--rounds 8] [--open 3000] [--seed 100]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks  # noqa: E402
from geo import World  # noqa: E402


def key_of(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fingerprint(item):
    return json.dumps({"state": item.state, "questions": item.questions}, sort_keys=True, ensure_ascii=True)


def base_tasks(world):
    return (tasks.geo_tasks(world) + tasks.math_tasks() + tasks.json_tasks() + tasks.time_tasks()
            + tasks.language_tasks() + tasks.knowledge_tasks() + tasks.option_scaling_tasks()
            + tasks.long_state_tasks() + tasks.logic_tasks() + tasks.text_tasks())


class SeededRandom(random.Random):
    """`random.Random(k)` as the generators call it, moved to another seed."""

    offset = 0

    def __init__(self, seed=None):
        super().__init__(None if seed is None else seed + SeededRandom.offset)


def regenerated(world, rounds, seed):
    """Every base family's items over `rounds` reseedings, less the benchmark's."""
    held_out = {fingerprint(i) for t in base_tasks(world) for i in t.items}
    original = tasks.random.Random
    tasks.random.Random = SeededRandom
    try:
        pool = {}
        for r in range(rounds):
            SeededRandom.offset = seed + 1000 * r
            for t in base_tasks(world):
                seen = pool.setdefault(t.name, {})
                for i in t.items:
                    f = fingerprint(i)
                    if f not in held_out and f not in seen:
                        seen[f] = {"body": i.request(), "gold": {q: key_of(g) for q, g in i.gold.items()}}
    finally:
        tasks.random.Random = original
    return {name: list(items.values()) for name, items in pool.items()}


# ---------------------------------------------------------------- open states

PRODUCTS = ["headphones", "a standing desk", "the mobile app", "a winter coat", "the coffee machine", "a mattress",
            "running shoes", "the router", "a toaster", "the tablet", "a car seat", "the subscription"]
ISSUES = ["arrived broken", "never arrived", "stopped working after a week", "is the wrong size", "was charged twice",
          "keeps disconnecting", "smells of chemicals", "is missing parts", "works perfectly", "exceeded my expectations",
          "was delivered a day early", "looks exactly like the photos"]
TONES = ["I'm furious.", "Please help.", "Not a big deal, just letting you know.", "This is the third time.", "Thanks in advance!",
         "I want a refund today.", "Honestly impressed.", "Could you look into it when you have a moment?", "", "Unacceptable."]
NAMES = ["Priya", "Tom", "Ayşe", "Marcus", "Lena", "Kenji", "Fatima", "Diego", "Nora", "Sam"]
DEPARTMENTS = ["billing", "shipping", "technical", "returns", "sales"]
SEVERITIES = ["low", "medium", "high", "critical"]
SERVICES = ["payments-api", "search", "login", "checkout", "image-cdn", "notifications", "reporting"]
SYMPTOMS = ["latency above 2 s", "5xx errors at 3%", "a full disk on one node", "certificate expiry in 2 days",
            "a memory leak", "duplicate emails sent", "stale cache entries", "a dropped database connection"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Santiago", "Warsaw", "Jakarta", "Denver"]
JOBS = ["engineer", "nurse", "teacher", "driver", "chef", "analyst", "designer", "pilot"]


def support_state(rng):
    name, product, issue, tone = rng.choice(NAMES), rng.choice(PRODUCTS), rng.choice(ISSUES), rng.choice(TONES)
    order = rng.randint(1000, 99999)
    return f"From {name}: My order #{order} for {product} {issue}. {tone}".strip()


def review_state(rng):
    stars = rng.randint(1, 5)
    product, issue = rng.choice(PRODUCTS), rng.choice(ISSUES)
    detail = rng.choice(["Battery life is", "Build quality is", "Customer service was", "Setup was", "Value for money is"])
    judgement = rng.choice(["excellent", "fine", "poor", "surprisingly good", "disappointing"])
    return {"product": product, "stars": stars, "text": f"{product.capitalize()} {issue}. {detail} {judgement}."}


def incident_state(rng):
    service, symptom = rng.choice(SERVICES), rng.choice(SYMPTOMS)
    minutes = rng.choice([5, 20, 45, 120, 480])
    users = rng.choice([0, 12, 300, 4000, 150000])
    return {"service": service, "symptom": symptom, "minutes": minutes, "users_affected": users,
            "note": rng.choice(["Rollback in progress.", "No owner assigned yet.", "Mitigated.", "Customers are reporting it.", ""])}


def profile_state(rng):
    name, city, job = rng.choice(NAMES), rng.choice(CITIES), rng.choice(JOBS)
    age = rng.randint(19, 70)
    years = rng.randint(0, min(45, age - 18))
    return {"name": name, "age": age, "city": city, "occupation": job, "years_in_role": years,
            "remote": rng.choice([True, False])}


COUNTRIES = {
    "Africa": ["Kenya", "Nigeria", "Egypt", "Morocco", "Ghana", "Ethiopia", "Tanzania", "Senegal", "Zambia", "Tunisia"],
    "Asia": ["Japan", "Vietnam", "Nepal", "Mongolia", "Thailand", "Iran", "Pakistan", "Malaysia", "South Korea", "Uzbekistan"],
    "Europe": ["Portugal", "Norway", "Hungary", "Greece", "Ireland", "Austria", "Croatia", "Finland", "Belgium", "Estonia"],
    "North America": ["Canada", "Mexico", "Cuba", "Guatemala", "Panama", "Honduras", "Jamaica", "Costa Rica"],
    "South America": ["Peru", "Colombia", "Uruguay", "Bolivia", "Ecuador", "Paraguay", "Venezuela", "Guyana"],
    "Oceania": ["New Zealand", "Fiji", "Samoa", "Papua New Guinea", "Tonga", "Vanuatu"],
}
ANIMALS = {
    "mammal": ["otter", "walrus", "giraffe", "hedgehog", "bat", "orca", "camel", "lemur", "moose", "platypus"],
    "bird": ["heron", "puffin", "kiwi", "falcon", "emu", "flamingo", "swan", "toucan"],
    "reptile": ["gecko", "iguana", "cobra", "tortoise", "chameleon", "monitor lizard"],
    "fish": ["salmon", "seahorse", "eel", "barracuda", "manta ray", "cod"],
    "insect": ["dragonfly", "termite", "mantis", "firefly", "weevil", "hornet"],
    "amphibian": ["axolotl", "newt", "tree frog", "salamander", "toad"],
}
FACTS = [
    ("Water boils at 100 degrees Celsius at sea level.", True), ("The Sahara is the largest hot desert on Earth.", True),
    ("Sound travels faster than light.", False), ("A hexagon has six sides.", True), ("The Pacific is the smallest ocean.", False),
    ("Venus is the closest planet to the Sun.", False), ("Copper conducts electricity.", True), ("Spiders are insects.", False),
    ("The Nile flows into the Mediterranean Sea.", True), ("Bats are blind.", False), ("Mount Everest is in the Andes.", False),
    ("Honey never spoils.", True), ("Humans have four lungs.", False), ("Lightning is hotter than the surface of the Sun.", True),
    ("The Great Wall is in India.", False), ("Octopuses have three hearts.", True), ("Gold is a magnetic metal.", False),
    ("A leap year has 366 days.", True), ("The Amazon is in Africa.", False), ("Ice is less dense than liquid water.", True),
    ("Penguins live at the North Pole.", False), ("The human skeleton has over 200 bones.", True), ("Mars has two moons.", True),
    ("Tomatoes are vegetables, botanically.", False), ("Sharks are mammals.", False), ("Helium is lighter than air.", True),
]
PHRASES = {
    "English": ["The train leaves at nine tomorrow.", "She painted the fence green.", "Could you pass the salt?", "It rained all night."],
    "Spanish": ["El tren sale a las nueve mañana.", "Ella pintó la cerca de verde.", "¿Me pasas la sal?", "Llovió toda la noche."],
    "French": ["Le train part à neuf heures demain.", "Elle a peint la clôture en vert.", "Peux-tu me passer le sel ?", "Il a plu toute la nuit."],
    "German": ["Der Zug fährt morgen um neun.", "Sie hat den Zaun grün gestrichen.", "Kannst du mir das Salz reichen?", "Es hat die ganze Nacht geregnet."],
    "Italian": ["Il treno parte alle nove domani.", "Ha dipinto la recinzione di verde.", "Mi passi il sale?", "Ha piovuto tutta la notte."],
    "Portuguese": ["O trem sai às nove amanhã.", "Ela pintou a cerca de verde.", "Pode me passar o sal?", "Choveu a noite toda."],
    "Dutch": ["De trein vertrekt morgen om negen uur.", "Ze heeft het hek groen geschilderd.", "Kun je me het zout aangeven?", "Het regende de hele nacht."],
    "Turkish": ["Tren yarın dokuzda kalkıyor.", "Çiti yeşile boyadı.", "Tuzu uzatır mısın?", "Bütün gece yağmur yağdı."],
    "Indonesian": ["Kereta berangkat jam sembilan besok.", "Dia mengecat pagar menjadi hijau.", "Bisa tolong ambilkan garam?", "Hujan turun sepanjang malam."],
    "Swahili": ["Treni inaondoka saa tatu kesho.", "Alipaka uzio rangi ya kijani.", "Unaweza kunipa chumvi?", "Mvua ilinyesha usiku kucha."],
    "Polish": ["Pociąg odjeżdża jutro o dziewiątej.", "Pomalowała płot na zielono.", "Podasz mi sól?", "Padało całą noc."],
    "Hindi": ["ट्रेन कल नौ बजे निकलती है।", "उसने बाड़ को हरा रंग दिया।", "क्या आप नमक दे सकते हैं?", "रात भर बारिश होती रही।"],
}
SNIPPETS = {
    "python": ["def area(r):\n    return 3.14159 * r ** 2", "items = [x * 2 for x in range(10) if x % 3]", "with open('log.txt') as f:\n    lines = f.readlines()",
               "print(1 / 0)", "raise ValueError('bad input')", "d = {}\nprint(d['missing'])", "import math\nprint(math.sqrt(16))"],
    "javascript": ["const area = r => Math.PI * r ** 2;", "const items = [1, 2, 3].map(x => x * 2);", "fetch(url).then(r => r.json());",
                   "null.length;", "throw new Error('bad input');", "JSON.parse('{bad json');", "console.log([1, 2, 3].includes(2));"],
    "rust": ["fn area(r: f64) -> f64 { std::f64::consts::PI * r * r }", "let items: Vec<i32> = (0..10).filter(|x| x % 3 != 0).collect();",
             "let v: Vec<i32> = Vec::new();\nlet x = v[3];", "panic!(\"bad input\");", "let n: u8 = 255;\nlet m = n.checked_add(1).unwrap();", "println!(\"{}\", 16f64.sqrt());"],
    "go": ["func area(r float64) float64 { return math.Pi * r * r }", "items := make([]int, 0, 10)", "var m map[string]int\nm[\"a\"] = 1",
           "panic(\"bad input\")", "fmt.Println(len(\"hello\"))", "x := []int{1, 2, 3}\nfmt.Println(x[5])"],
    "sql": ["SELECT name, total FROM orders WHERE total > 100 ORDER BY total DESC;", "UPDATE users SET active = 0 WHERE last_login < '2024-01-01';",
            "CREATE TABLE t (id INT PRIMARY KEY, name TEXT);", "SELECT COUNT(*) FROM events GROUP BY day;"],
}
SPAM = ["Congratulations! You have been selected for a free cruise. Click now to claim.", "URGENT: your account will be closed unless you verify your password at this link.",
        "Make $5000 a week from home, no experience needed!!!", "You won a lottery you never entered. Send the processing fee to collect.",
        "Hot singles in your area are waiting. Reply YES.", "Limited offer: cheap meds without prescription, discreet shipping."]
HAM = ["Can we move the standup to 10:30 tomorrow?", "The invoice for March is attached; let me know if anything is off.",
       "Reminder: dentist appointment on Thursday at 2 pm.", "Your package was delivered to the front desk.",
       "Thanks for the feedback on the draft, I'll revise section 3.", "Are we still on for dinner on Saturday?"]
TOPICS = {
    "sports": ["The home side equalised in stoppage time after a corner.", "The sprinter set a national record in the semi-final.", "The coach rested three starters before the final."],
    "finance": ["Shares fell 4% after the earnings call missed guidance.", "The central bank held rates and signalled a cut in spring.", "Bond yields rose for a third straight session."],
    "science": ["The probe returned samples from the asteroid's surface.", "Researchers sequenced the genome of a deep-sea worm.", "The trial showed the vaccine cut infections by half."],
    "politics": ["The senate passed the budget after a late amendment.", "Turnout in the regional election reached 61%.", "The minister resigned over the leaked memo."],
    "technology": ["The chipmaker unveiled a 3-nanometre process.", "The update broke login for users on older phones.", "The startup open-sourced its database engine."],
    "health": ["Doctors recommend two hours of exercise a week for adults.", "The clinic reported a rise in flu cases this month.", "A new guideline lowers the screening age to 45."],
}


def knowledge_state(rng):
    kind = rng.choice(["continent", "animal", "fact", "fact"])
    if kind == "continent":
        continent = rng.choice(list(COUNTRIES))
        country = rng.choice(COUNTRIES[continent])
        return {"kind": "country", "text": f"{country}"}
    if kind == "animal":
        cls = rng.choice(list(ANIMALS))
        return {"kind": "animal", "text": rng.choice(ANIMALS[cls])}
    text, _ = rng.choice(FACTS)
    return {"kind": "statement", "text": text}


def language_state(rng):
    lang = rng.choice(list(PHRASES))
    return rng.choice(PHRASES[lang])


def code_state(rng):
    lang = rng.choice(list(SNIPPETS))
    return {"code": rng.choice(SNIPPETS[lang])}


def message_state(rng):
    return rng.choice(SPAM + HAM)


def topic_state(rng):
    topic = rng.choice(list(TOPICS))
    return rng.choice(TOPICS[topic])


OPEN_FAMILIES = {
    "open_knowledge": (knowledge_state, {
        "continent": tasks.choice("If the text names a country, on which continent is it?", {c: None for c in COUNTRIES}),
        "animal_class": tasks.choice("If the text names an animal, what kind is it?", {c: None for c in ANIMALS}),
        "true": tasks.noul("If the text is a statement, is it true?"),
    }),
    "open_language": (language_state, {
        "language": tasks.choice("Which language is the text written in?", {l: None for l in PHRASES}),
        "question": tasks.noul("Is the text a question?"),
    }),
    "open_code": (code_state, {
        "language": tasks.choice("Which programming language is this?", {l: None for l in SNIPPETS}),
        "raises": tasks.noul("Would running this code raise an error or panic?"),
    }),
    "open_message": (message_state, {
        "spam": tasks.noul("Is this message spam?"),
        "urgency": tasks.score("How urgent is this message?", ["not urgent", "soon", "today", "immediately"]),
    }),
    "open_topic": (topic_state, {
        "topic": tasks.choice("What is the topic of the sentence?", {t: None for t in TOPICS}),
        "sentiment": tasks.choice("What is the sentiment of the sentence?", {"negative": None, "neutral": None, "positive": None}),
    }),
    "open_support": (support_state, {
        "department": tasks.choice("Which department should handle this message?", {d: None for d in DEPARTMENTS}),
        "urgency": tasks.score("How urgent is this?", ["can wait", "this week", "today", "right now"]),
        "angry": tasks.noul("Is the customer angry?"),
        "refund": tasks.noul("Is the customer asking for money back?"),
    }),
    "open_review": (review_state, {
        "sentiment": tasks.choice("What is the sentiment of the review?", {"negative": None, "neutral": None, "positive": None}),
        "recommend": tasks.noul("Would the reviewer recommend the product?"),
        "quality": tasks.score("How does the reviewer rate the product's quality?", ["poor", "fair", "good", "excellent"]),
    }),
    "open_incident": (incident_state, {
        "severity": tasks.choice("What severity should this incident have?", {s: None for s in SEVERITIES}),
        "page": tasks.noul("Should the on-call engineer be paged now?"),
        "customer_facing": tasks.noul("Does this affect customers?"),
    }),
    "open_profile": (profile_state, {
        "senior": tasks.noul("Has this person spent more than ten years in their role?"),
        "continent": tasks.choice("On which continent is this person's city?",
                                  {c: None for c in ["Africa", "Asia", "Europe", "North America", "South America"]}),
        "adult_under_30": tasks.noul("Is this person under 30?"),
    }),
}


PLACES_FILE = "ne_110m_populated_places.geojson"


def natural_earth_knowledge(world, seed):
    """Country-to-continent and capital items from Natural Earth: every country the
    benchmark's geography already relies on, asked by name, and the admin-0 capitals
    of the populated-places layer. Answers come from the data, not a model."""
    import urllib.request
    from geo import CACHE, SOURCE
    rng = random.Random(seed)
    countries = sorted({(s.properties["ADMIN"], s.properties["CONTINENT"]) for s in world.countries})
    continents = sorted({c for _, c in countries})
    continent_q = tasks.choice("On which continent is this country?", {c: None for c in continents})
    rows = {"country_continent": []}
    for name, continent in countries:
        rows["country_continent"].append({"body": {"state": {"country": name}, "questions": {"continent": continent_q}},
                                          "gold": {"continent": continent}})
    path = CACHE / PLACES_FILE
    try:
        if not path.is_file():
            with urllib.request.urlopen(f"{SOURCE}/{PLACES_FILE}", timeout=60) as response:
                path.write_bytes(response.read())
        places = json.loads(path.read_text())["features"]
    except Exception as e:  # noqa: BLE001 - the capitals are optional
        print(f"capitals skipped: {e}")
        return rows
    capitals = {}
    for f in places:
        p = f["properties"]
        if p.get("FEATURECLA") == "Admin-0 capital" and p.get("ADM0NAME") and p.get("NAME"):
            capitals.setdefault(p["ADM0NAME"], p["NAME"])
    capital_q = tasks.noul("Is the city the capital of the country?")
    cities = sorted(set(capitals.values()))
    rows["capital_pairs"] = []
    # The benchmark's own capitals stay unseen: its countries are left out entirely.
    benchmark = {c.lower() for c in tasks.CAPITALS}
    for country, city in sorted(capitals.items()):
        if country.lower() in benchmark or any(c.lower() in country.lower() or country.lower() in c.lower() for c in benchmark):
            continue
        rows["capital_pairs"].append({"body": {"state": {"country": country, "city": city}, "questions": {"capital": capital_q}}, "gold": {"capital": "true"}})
        other = rng.choice([c for c in cities if c != city])
        rows["capital_pairs"].append({"body": {"state": {"country": country, "city": other}, "questions": {"capital": capital_q}}, "gold": {"capital": "false"}})
    return rows


def open_states(count, seed):
    rng = random.Random(seed)
    pool = {}
    for family, (make, questions) in OPEN_FAMILIES.items():
        # A family with fewer distinct states than asked for stops once a long run of
        # draws finds nothing new.
        seen = {}
        misses = 0
        while len(seen) < count and misses < 2000:
            state = make(rng)
            f = json.dumps(state, sort_keys=True, ensure_ascii=True)
            if f in seen:
                misses += 1
                continue
            misses = 0
            seen[f] = {"body": {"state": state, "questions": questions}}
        pool[family] = list(seen.values())
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rounds", type=int, default=8, help="reseedings of every benchmark family")
    parser.add_argument("--open", type=int, default=3000, help="states per open family")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    world = World()
    pool = regenerated(world, args.rounds, args.seed)
    held_out = {fingerprint(i) for t in base_tasks(world) for i in t.items}
    for family, rows in natural_earth_knowledge(world, args.seed).items():
        pool[family] = [r for r in rows if json.dumps(r["body"], sort_keys=True, ensure_ascii=True) not in held_out]
    pool.update(open_states(args.open, args.seed))
    total = 0
    for family, rows in sorted(pool.items()):
        if not rows:
            print(f"{family:26s} nothing new (finite content)")
            continue
        with (args.out / f"{family}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{family:26s} {len(rows):6d} {'judged by the teacher' if 'gold' not in rows[0] else ''}")
    print(f"{total} requests in {args.out}")


if __name__ == "__main__":
    main()
