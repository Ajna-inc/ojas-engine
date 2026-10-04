"""The evaluation tasks: decision requests with known answers.

A task is a list of items. An item is one request (`state` and `questions`) and
the right answer to each question: an option key for a `choice`, a level index for
a `score`, and a boolean for a `noul`. Generated tasks compute their answers;
written tasks carry them. Every generator is seeded, so a task is the same on
every run.

Robustness tasks rewrite items of other tasks without changing their answer (the
options reordered, the instructions reworded, the state's keys reordered); each
rewritten item names the item it rewrites, so a model's consistency can be scored.
"""

import calendar
import datetime
import random
from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    state: object
    questions: dict
    gold: dict
    # For a robustness variant, the id of the item it rewrites; the two should agree.
    variant_of: str = None

    def request(self):
        return {"state": self.state, "questions": self.questions}


@dataclass
class Task:
    name: str
    category: str
    description: str
    items: list = field(default_factory=list)


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions, criteria=None):
    q = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


def balanced(rng, make, want_true, want_false, attempts=100000):
    """Draw from `make()` (returning `(item, label)` or `None`) until `want_true`
    true and `want_false` false items are collected."""
    yes, no = [], []
    for _ in range(attempts):
        if len(yes) >= want_true and len(no) >= want_false:
            break
        made = make()
        if made is None:
            continue
        item, label = made
        bucket, want = (yes, want_true) if label else (no, want_false)
        if len(bucket) < want:
            bucket.append(item)
    items = yes + no
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------- geography

def geo_tasks(world):
    rng = random.Random(1)
    land_q = noul("Is there land at these coordinates (as opposed to ocean or sea)?")

    def land_item():
        lat, lon = rng.uniform(-75, 80), rng.uniform(-180, 180)
        if not world.stable(lat, lon, world.is_land):
            return None
        is_land = world.is_land(lat, lon)
        state = {"latitude": round(lat, 2), "longitude": round(lon, 2)}
        return Item(f"land-{lat:.2f},{lon:.2f}", state, {"land": land_q}, {"land": is_land}), is_land

    land = Task("geo_land_water", "geography", "Whether a latitude/longitude is on land or at sea.",
                balanced(rng, land_item, 60, 60))

    continents = ["Africa", "Antarctica", "Asia", "Europe", "North America", "Oceania", "South America"]
    continent_q = choice("On which continent are these coordinates?", {c: None for c in continents})
    by_continent = {c: [] for c in continents}
    country_items = []
    countries_seen = set()
    country_names = sorted({s.properties["ADMIN"] for s in world.countries})
    for _ in range(200000):
        if all(len(v) >= 10 for v in by_continent.values()) and len(country_items) >= 40:
            break
        lat, lon = rng.uniform(-85, 80), rng.uniform(-180, 180)
        here = world.country(lat, lon)
        if here is None or not world.stable(lat, lon, world.country, margin=1.0):
            continue
        name, cont = here
        state = {"latitude": round(lat, 2), "longitude": round(lon, 2)}
        if len(by_continent[cont]) < 10:
            by_continent[cont].append(Item(f"continent-{lat:.2f},{lon:.2f}", state, {"continent": continent_q}, {"continent": cont}))
        if len(country_items) < 40 and name not in countries_seen:
            others = rng.sample([c for c in country_names if c != name], 3)
            options = others + [name]
            rng.shuffle(options)
            q = choice("Which country are these coordinates in?", {o: None for o in options})
            country_items.append(Item(f"country-{lat:.2f},{lon:.2f}", state, {"country": q}, {"country": name}))
            countries_seen.add(name)
    continent_items = [i for v in by_continent.values() for i in v]
    rng.shuffle(continent_items)
    continent = Task("geo_continent", "geography", "The continent at a latitude/longitude, of seven.", continent_items)
    country = Task("geo_country", "geography", "The country at a latitude/longitude, of four named.", country_items)

    hemi_q = choice("Which hemisphere are these coordinates in?",
                    {"northern": "north of the equator", "southern": "south of the equator"})
    hemi_items = []
    for i in range(40):
        lat = rng.choice([1, -1]) * rng.uniform(5, 80)
        state = {"latitude": round(lat, 2), "longitude": round(rng.uniform(-180, 180), 2)}
        hemi_items.append(Item(f"hemisphere-{i}", state, {"hemisphere": hemi_q}, {"hemisphere": "northern" if lat > 0 else "southern"}))
    hemisphere = Task("geo_hemisphere", "geography", "Northern or southern hemisphere from a latitude.", hemi_items)
    return [land, continent, country, hemisphere]


# ---------------------------------------------------------------- arithmetic

def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def math_tasks():
    rng = random.Random(2)
    compare_q = choice("Which number is larger?", {"a": "the number a", "b": "the number b"})
    compare = []
    for i in range(50):
        a = rng.choice([rng.randint(1, 10000), round(rng.uniform(0, 100), 2)])
        b = a
        while b == a:
            b = round(a * rng.uniform(0.5, 1.5), 2) if rng.random() < 0.5 else rng.randint(1, 10000)
        compare.append(Item(f"compare-{i}", {"a": a, "b": b}, {"larger": compare_q}, {"larger": "a" if a > b else "b"}))

    prime_q = noul("Is n a prime number?")
    primes = [n for n in range(2, 300) if is_prime(n)]
    composites = [n for n in range(4, 300) if not is_prime(n)]
    prime = [Item(f"prime-{n}", {"n": n}, {"prime": prime_q}, {"prime": is_prime(n)})
             for n in rng.sample(primes, 25) + rng.sample(composites, 25)]
    rng.shuffle(prime)

    parity_q = choice("Is the number even or odd?", {"even": None, "odd": None})
    parity = [Item(f"parity-{i}", {"number": n}, {"parity": parity_q}, {"parity": "even" if n % 2 == 0 else "odd"})
              for i, n in enumerate(rng.randint(1, 10000) for _ in range(40))]

    sum_q = noul("Is the claimed result of the calculation correct?")
    sums = []
    for i in range(50):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["+", "-"])
        right = a + b if op == "+" else a - b
        claimed = right if i % 2 == 0 else right + rng.choice([-10, -2, -1, 1, 2, 10])
        sums.append(Item(f"sum-{i}", {"calculation": f"{a} {op} {b}", "claimed_result": claimed},
                         {"correct": sum_q}, {"correct": claimed == right}))

    levels = ["below 10", "10 to 99", "100 to 999", "1000 or more"]
    magnitude_q = score("How large is the number?", levels)
    magnitude = []
    for i in range(40):
        level = i % 4
        n = rng.randint([1, 10, 100, 1000][level], [9, 99, 999, 99999][level])
        magnitude.append(Item(f"magnitude-{i}", {"number": n}, {"size": magnitude_q}, {"size": level}))
    rng.shuffle(magnitude)
    return [
        Task("math_compare", "arithmetic", "Which of two numbers is larger.", compare),
        Task("math_prime", "arithmetic", "Whether a number below 300 is prime.", prime),
        Task("math_parity", "arithmetic", "Whether a number is even or odd.", parity),
        Task("math_check", "arithmetic", "Whether a claimed sum or difference is right.", sums),
        Task("math_magnitude", "arithmetic", "The order of magnitude of a number, on four levels.", magnitude),
    ]


# ---------------------------------------------------------------- structured data

PRODUCTS = ["keyboard", "mouse", "monitor", "cable", "headphones", "webcam", "desk lamp", "notebook", "charger", "speaker"]


def json_tasks():
    rng = random.Random(3)
    total_q = noul("Is the order total above 100 dollars?")

    def order():
        items = [{"name": rng.choice(PRODUCTS), "price": round(rng.uniform(3, 80), 2), "quantity": rng.randint(1, 3)}
                 for _ in range(rng.randint(1, 4))]
        total = sum(x["price"] * x["quantity"] for x in items)
        if abs(total - 100) < 5:
            return None
        state = {"order_id": f"A{rng.randint(1000, 9999)}", "currency": "USD", "items": items}
        return Item(f"order-{state['order_id']}-{len(items)}", state, {"above": total_q}, {"above": total > 100}), total > 100

    orders = balanced(rng, order, 25, 25)

    plans = ["free", "pro", "enterprise"]
    plan_q = choice("Which plan is the user on?", {p: None for p in plans})
    names = ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gita", "Hugo", "Ines", "Jon"]
    lookup = []
    for i in range(40):
        plan = plans[i % 3]
        record = {"user": {"name": rng.choice(names), "id": 5000 + i,
                           "account": {"created": f"202{rng.randint(0, 4)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
                                       "plan": plan, "seats": rng.randint(1, 50)}},
                  "last_login_days": rng.randint(0, 90)}
        lookup.append(Item(f"plan-{i}", record, {"plan": plan_q}, {"plan": plan}))
    rng.shuffle(lookup)

    count_levels = ["0", "1", "2", "3", "4", "5 or more"]
    count_q = score("How many items are in the cart?", count_levels)
    counts = []
    for i in range(36):
        n = i % 6 if i % 6 < 5 else rng.randint(5, 8)
        cart = {"cart": [{"sku": f"SKU-{rng.randint(100, 999)}", "name": rng.choice(PRODUCTS)} for _ in range(n)]}
        counts.append(Item(f"count-{i}", cart, {"count": count_q}, {"count": min(n, 5)}))
    rng.shuffle(counts)
    return [
        Task("json_order_total", "structured", "Whether an order's total, from item prices and quantities, is above 100.", orders),
        Task("json_lookup", "structured", "A field read from a nested record.", lookup),
        Task("json_count", "structured", "How many entries a list holds, on six levels.", counts),
    ]


# ---------------------------------------------------------------- dates and times

def time_tasks():
    rng = random.Random(4)
    weekend_q = noul("Does this date fall on a weekend (Saturday or Sunday)?")
    weekend = []
    start = datetime.date(2015, 1, 1)
    for i in range(50):
        while True:
            d = start + datetime.timedelta(days=rng.randint(0, 4000))
            if (d.weekday() >= 5) == (i % 2 == 0):
                break
        weekend.append(Item(f"weekend-{d}", {"date": d.isoformat()}, {"weekend": weekend_q}, {"weekend": d.weekday() >= 5}))
    rng.shuffle(weekend)

    day_q = choice("Which part of the day is this time?", {
        "morning": "from 05:00 to 11:59", "afternoon": "from 12:00 to 16:59",
        "evening": "from 17:00 to 20:59", "night": "from 21:00 to 04:59"})
    def part(h):
        return "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 21 else "night"
    day = []
    for i in range(40):
        h, m = rng.randint(0, 23), rng.randint(0, 59)
        day.append(Item(f"time-{h:02d}{m:02d}-{i}", {"time": f"{h:02d}:{m:02d}"}, {"part": day_q}, {"part": part(h)}))

    month_q = noul("Does this month have 31 days?")

    def month():
        y, mo = rng.randint(2000, 2030), rng.randint(1, 12)
        long = calendar.monthrange(y, mo)[1] == 31
        return Item(f"month-{y}-{mo}", {"month": f"{calendar.month_name[mo]} {y}"}, {"long": month_q}, {"long": long}), long

    months = balanced(rng, month, 18, 18)
    return [
        Task("date_weekend", "dates", "Whether a calendar date is a Saturday or Sunday.", weekend),
        Task("time_of_day", "dates", "Morning, afternoon, evening or night from a clock time.", day),
        Task("month_length", "dates", "Whether a month has 31 days.", months),
    ]


# ---------------------------------------------------------------- language

SENTIMENT = [
    ("I absolutely love this phone, the battery lasts for days.", "positive"),
    ("The delivery was fast and the packaging was perfect.", "positive"),
    ("Best customer service I have ever dealt with.", "positive"),
    ("This exceeded every expectation I had.", "positive"),
    ("The new update made the app so much smoother, great job.", "positive"),
    ("I'm thrilled with how the room turned out.", "positive"),
    ("Five stars, would buy again without hesitation.", "positive"),
    ("The staff were friendly and the food was delicious.", "positive"),
    ("What a fantastic concert, the band was on fire.", "positive"),
    ("Setup took two minutes and everything just worked.", "positive"),
    ("Honestly one of the most comfortable chairs I've owned.", "positive"),
    ("The course was clear, practical and fun.", "positive"),
    ("The product broke after two days. Total waste of money.", "negative"),
    ("I waited an hour on hold and nobody helped me.", "negative"),
    ("The hotel room was dirty and smelled awful.", "negative"),
    ("Terrible app, it crashes every time I open it.", "negative"),
    ("I'm really disappointed with the quality of this jacket.", "negative"),
    ("They charged me twice and refuse to refund me.", "negative"),
    ("The movie was boring and far too long.", "negative"),
    ("Worst purchase I have made this year.", "negative"),
    ("The food arrived cold and the order was wrong.", "negative"),
    ("I regret signing up, the service is unreliable.", "negative"),
    ("The instructions were confusing and parts were missing.", "negative"),
    ("Never again. Rude staff and overpriced drinks.", "negative"),
    ("The package arrived on Tuesday.", "neutral"),
    ("The store opens at nine in the morning.", "neutral"),
    ("I bought the blue version of the shirt.", "neutral"),
    ("The meeting has been moved to the second floor.", "neutral"),
    ("The manual is available in English and Spanish.", "neutral"),
    ("My order number is 48213.", "neutral"),
    ("The train leaves from platform four.", "neutral"),
    ("The software update is version 3.2.", "neutral"),
    ("I am writing to ask about the return policy.", "neutral"),
    ("The box contains a charger and a cable.", "neutral"),
    ("The museum is closed on Mondays.", "neutral"),
    ("We moved to a new office last month.", "neutral"),
]

SENTIMENT_MULTILINGUAL = [
    ("es", "Me encanta este restaurante, la comida es increíble.", "positive"),
    ("es", "El producto llegó roto y nadie responde mis correos.", "negative"),
    ("es", "La tienda abre a las diez de la mañana.", "neutral"),
    ("es", "El servicio fue excelente y muy rápido.", "positive"),
    ("es", "Qué decepción, no funciona nada.", "negative"),
    ("es", "El paquete pesa dos kilos.", "neutral"),
    ("fr", "J'adore cette application, elle est vraiment pratique.", "positive"),
    ("fr", "Le colis est arrivé en retard et abîmé.", "negative"),
    ("fr", "La réunion commence à quatorze heures.", "neutral"),
    ("fr", "Un séjour merveilleux, je recommande vivement.", "positive"),
    ("fr", "Service client horrible, je suis très déçu.", "negative"),
    ("fr", "Le magasin se trouve rue Victor Hugo.", "neutral"),
    ("de", "Das Essen war hervorragend und der Service sehr freundlich.", "positive"),
    ("de", "Die Lieferung war beschädigt und unvollständig.", "negative"),
    ("de", "Der Zug fährt um acht Uhr ab.", "neutral"),
    ("de", "Ich bin begeistert von diesem Kopfhörer.", "positive"),
    ("de", "Schlechteste Erfahrung seit langem.", "negative"),
    ("de", "Das Formular hat drei Seiten.", "neutral"),
    ("pt", "Adorei o hotel, os quartos são lindos.", "positive"),
    ("pt", "O aplicativo trava o tempo todo, péssimo.", "negative"),
    ("pt", "A loja fica no segundo andar.", "neutral"),
    ("hi", "यह फ़ोन बहुत शानदार है, मुझे बहुत पसंद आया।", "positive"),
    ("hi", "सेवा बहुत खराब थी और कोई मदद नहीं मिली।", "negative"),
    ("hi", "दुकान सुबह दस बजे खुलती है।", "neutral"),
    ("ja", "このカフェは最高です。コーヒーがとても美味しい。", "positive"),
    ("ja", "商品が壊れて届きました。とても残念です。", "negative"),
    ("ja", "会議は午後三時に始まります。", "neutral"),
    ("zh", "这家酒店非常棒，服务很周到。", "positive"),
    ("zh", "质量太差了，用了一天就坏了。", "negative"),
    ("zh", "火车早上八点出发。", "neutral"),
]

ROUTING = [
    ("I was charged twice for my subscription this month.", "billing"),
    ("Can I get a refund for the order I cancelled?", "billing"),
    ("My invoice shows the wrong company name.", "billing"),
    ("Why did my monthly price go up?", "billing"),
    ("The payment failed but money left my account.", "billing"),
    ("I need a receipt for my last purchase for tax purposes.", "billing"),
    ("Please update the credit card on file.", "billing"),
    ("I was billed after I cancelled my plan.", "billing"),
    ("My package has not arrived and tracking hasn't updated in a week.", "shipping"),
    ("Can you change the delivery address for order 5521?", "shipping"),
    ("The courier left my parcel at the wrong house.", "shipping"),
    ("Do you ship to Canada?", "shipping"),
    ("My order arrived with a missing item.", "shipping"),
    ("How long does express delivery take?", "shipping"),
    ("The box was crushed when it was delivered.", "shipping"),
    ("I want to schedule a delivery for next Monday.", "shipping"),
    ("The app crashes when I upload a photo.", "technical"),
    ("I get an error 500 when saving my settings.", "technical"),
    ("The website won't load on my phone.", "technical"),
    ("Sync between my laptop and phone stopped working.", "technical"),
    ("The export button does nothing when I click it.", "technical"),
    ("Notifications are not showing up on Android.", "technical"),
    ("The API returns a timeout for every request.", "technical"),
    ("Video calls keep freezing after a few minutes.", "technical"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("How do I change the email address on my profile?", "account"),
    ("Please delete my account and all my data.", "account"),
    ("I want to add a second user to my team.", "account"),
    ("My account was locked after too many login attempts.", "account"),
    ("Can I change my username?", "account"),
    ("How do I turn on two-factor authentication?", "account"),
    ("Someone else seems to have logged into my account.", "account"),
]

SPAM = [
    ("Congratulations! You have won a $1000 gift card. Click here to claim now!", True),
    ("URGENT: your bank account is suspended, verify your password at this link.", True),
    ("Earn $5000 a week working from home, no experience needed!!!", True),
    ("You are the lucky winner of a free iPhone, reply with your address.", True),
    ("Cheap watches, best prices, buy now, limited offer!!!", True),
    ("Dear friend, I am a prince and need help transferring 10 million dollars.", True),
    ("Your parcel is held at customs, pay a small fee here to release it.", True),
    ("Lose 20 pounds in 5 days with this one weird trick.", True),
    ("Act now! Your computer is infected, call this number immediately.", True),
    ("Claim your crypto airdrop before it expires, connect your wallet here.", True),
    ("Exclusive deal just for you: 90% off designer bags, today only!", True),
    ("Your account will be deleted unless you confirm your login details now.", True),
    ("Hi Sam, are we still on for lunch tomorrow at noon?", False),
    ("Attached is the report you asked for. Let me know if anything is missing.", False),
    ("Your order #4821 has shipped and should arrive Thursday.", False),
    ("Reminder: the team meeting moves to 3pm today.", False),
    ("Thanks for your help yesterday, the fix worked.", False),
    ("Can you review my pull request when you have a moment?", False),
    ("Mom says dinner is at seven on Sunday.", False),
    ("The library book you reserved is ready for pickup.", False),
    ("Here are the slides from this morning's presentation.", False),
    ("Your appointment with Dr. Lee is confirmed for March 3.", False),
    ("Could you send me the invoice for last month?", False),
    ("Happy birthday! Hope you have a great day.", False),
]

URGENCY = [
    ("Whenever you get a chance, could you update the logo on the about page?", 0),
    ("No rush, but it would be nice to have dark mode at some point.", 0),
    ("Just a suggestion for a future release: add CSV export.", 0),
    ("Small typo on the pricing page, fix it when convenient.", 0),
    ("Could you look into the slow report page sometime this week?", 1),
    ("Please send the updated contract before Friday.", 1),
    ("A few users mentioned the search is a bit slow, can we check this week?", 1),
    ("We need the new invoices ready for the end-of-week review.", 1),
    ("The client demo is this afternoon and the login page shows an error.", 2),
    ("Payroll has to be submitted by 5pm today and the export is failing.", 2),
    ("Our presentation starts in a few hours and the slides won't open.", 2),
    ("Please fix the broken checkout button before tonight's sale.", 2),
    ("The whole site is down and customers cannot pay right now!", 3),
    ("Production database is deleting records, stop it immediately!", 3),
    ("We are being charged thousands per minute by a runaway job, help now!", 3),
    ("Security breach in progress, attackers are logged in as admin!", 3),
]

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra",
    "Brazil": "Brasília", "India": "New Delhi", "Kenya": "Nairobi", "Egypt": "Cairo", "Mexico": "Mexico City",
    "Italy": "Rome", "Spain": "Madrid", "Turkey": "Ankara", "Argentina": "Buenos Aires", "South Korea": "Seoul",
    "Nigeria": "Abuja", "Norway": "Oslo", "Peru": "Lima", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Poland": "Warsaw", "Morocco": "Rabat", "Chile": "Santiago", "Sweden": "Stockholm", "Portugal": "Lisbon",
    "Pakistan": "Islamabad", "New Zealand": "Wellington", "South Africa": "Pretoria", "Switzerland": "Bern",
    "Indonesia": "Jakarta", "Greece": "Athens", "Ireland": "Dublin", "Colombia": "Bogotá", "Iran": "Tehran",
    "Ethiopia": "Addis Ababa", "Ukraine": "Kyiv", "Cuba": "Havana", "Finland": "Helsinki", "Philippines": "Manila",
}

ANIMALS = [
    ("whale", "mammal"), ("bat", "mammal"), ("dolphin", "mammal"), ("elephant", "mammal"), ("kangaroo", "mammal"),
    ("platypus", "mammal"), ("horse", "mammal"),
    ("penguin", "bird"), ("ostrich", "bird"), ("eagle", "bird"), ("owl", "bird"), ("parrot", "bird"), ("emu", "bird"),
    ("crocodile", "reptile"), ("snake", "reptile"), ("turtle", "reptile"), ("lizard", "reptile"), ("chameleon", "reptile"),
    ("shark", "fish"), ("salmon", "fish"), ("seahorse", "fish"), ("tuna", "fish"), ("eel", "fish"), ("goldfish", "fish"),
    ("frog", "amphibian"), ("salamander", "amphibian"), ("newt", "amphibian"), ("toad", "amphibian"), ("axolotl", "amphibian"),
    ("ant", "insect"), ("butterfly", "insect"), ("beetle", "insect"), ("bee", "insect"), ("dragonfly", "insect"), ("grasshopper", "insect"),
]

# (unit, size in grams or in centimetres)
MASSES = [("g", 1.0), ("kg", 1000.0), ("lb", 453.592), ("oz", 28.3495)]
LENGTHS = [("cm", 1.0), ("m", 100.0), ("km", 100000.0), ("inch", 2.54), ("foot", 30.48), ("mile", 160934.0)]


def language_tasks():
    rng = random.Random(5)
    sentiment_q = choice("What is the sentiment of the text?", {"positive": None, "negative": None, "neutral": None})
    sentiment = [Item(f"sentiment-{i}", text, {"sentiment": sentiment_q}, {"sentiment": label})
                 for i, (text, label) in enumerate(SENTIMENT)]
    multi = [Item(f"sentiment-{lang}-{i}", text, {"sentiment": sentiment_q}, {"sentiment": label})
             for i, (lang, text, label) in enumerate(SENTIMENT_MULTILINGUAL)]
    routing_q = choice("Which team should handle this message?", {
        "billing": "payments, charges, invoices, refunds", "shipping": "deliveries, parcels, couriers",
        "technical": "bugs, errors, crashes", "account": "login, profile, users, security of the account"})
    routing = [Item(f"routing-{i}", text, {"team": routing_q}, {"team": label}) for i, (text, label) in enumerate(ROUTING)]
    spam_q = noul("Is this message spam or a scam?")
    spam = [Item(f"spam-{i}", text, {"spam": spam_q}, {"spam": label}) for i, (text, label) in enumerate(SPAM)]
    urgency_q = score("How urgent is this request?", ["can wait", "this week", "today", "right now"])
    urgency = [Item(f"urgency-{i}", text, {"urgency": urgency_q}, {"urgency": level}) for i, (text, level) in enumerate(URGENCY)]
    for t in (sentiment, multi, routing, spam, urgency):
        rng.shuffle(t)
    return [
        Task("sentiment", "language", "Positive, negative or neutral sentiment of an English sentence.", sentiment),
        Task("sentiment_multilingual", "language", "Sentiment in Spanish, French, German, Portuguese, Hindi, Japanese and Chinese.", multi),
        Task("ticket_routing", "language", "The team a support message belongs to, of four.", routing),
        Task("spam", "language", "Whether a message is spam or a scam.", spam),
        Task("urgency", "language", "How urgent a request is, on four levels.", urgency),
    ]


# ---------------------------------------------------------------- knowledge

def knowledge_tasks():
    rng = random.Random(6)
    capital_q = noul("Is the city the capital of the country?")
    pairs = sorted(CAPITALS.items())
    capitals = []
    for country, city in pairs:
        capitals.append(Item(f"capital-{country}-true", {"country": country, "city": city}, {"capital": capital_q}, {"capital": True}))
        wrong = rng.choice([c for k, c in pairs if k != country])
        capitals.append(Item(f"capital-{country}-false", {"country": country, "city": wrong}, {"capital": capital_q}, {"capital": False}))
    rng.shuffle(capitals)

    classes = ["mammal", "bird", "reptile", "fish", "amphibian", "insect"]
    animal_q = choice("Which class of animal is this?", {c: None for c in classes})
    animals = [Item(f"animal-{name}", {"animal": name}, {"class": animal_q}, {"class": cls}) for name, cls in ANIMALS]
    rng.shuffle(animals)

    heavier_q = choice("Which is heavier?", {"first": "the first quantity", "second": "the second quantity"})
    longer_q = choice("Which is longer?", {"first": "the first quantity", "second": "the second quantity"})
    units = []
    for i in range(40):
        table, q = (MASSES, heavier_q) if i % 2 == 0 else (LENGTHS, longer_q)
        while True:
            (u1, b1), (u2, b2) = rng.sample(table, 2)
            v1 = rng.choice([1, 2, 3, 5, 10, 20, 50, 100, 250, 500])
            v2 = round(v1 * b1 / b2 * rng.choice([0.6, 0.8, 1.25, 1.6]), 1)
            if v2 > 0 and abs(v1 * b1 - v2 * b2) / max(v1 * b1, v2 * b2) > 0.1:
                break
        units.append(Item(f"units-{i}", {"first": f"{v1} {u1}", "second": f"{v2} {u2}"}, {"answer": q},
                          {"answer": "first" if v1 * b1 > v2 * b2 else "second"}))
    return [
        Task("capitals", "knowledge", "Whether a city is the capital of a country.", capitals),
        Task("animal_class", "knowledge", "The class of an animal, of six.", animals),
        Task("unit_compare", "knowledge", "Which of two masses or lengths in different units is larger.", units),
    ]


# ---------------------------------------------------------------- option count

def option_scaling_tasks():
    """The same question with 2 to all 39 countries as options: how accuracy falls
    as the choice widens."""
    rng = random.Random(7)
    countries = sorted(CAPITALS)
    tasks = []
    for n in (2, 5, 10, 20, len(countries)):
        items = []
        for country in rng.sample(countries, 30):
            options = [country] + rng.sample([c for c in countries if c != country], n - 1)
            rng.shuffle(options)
            q = choice("Which country has this city as its capital?", {o: None for o in options})
            items.append(Item(f"capital-of-{CAPITALS[country]}-{n}", {"city": CAPITALS[country]}, {"country": q}, {"country": country}))
        tasks.append(Task(f"options_{n:02d}", "option count", f"The country whose capital a city is, of {n}.", items))
    return tasks


# ---------------------------------------------------------------- long state

CITIES = ["Lisbon", "Oslo", "Lagos", "Lima", "Hanoi", "Perth", "Quito", "Riga", "Cairo", "Dakar", "Seoul", "Turin",
          "Leeds", "Austin", "Denver", "Osaka", "Pune", "Accra", "Bergen", "Cork", "Graz", "Malmo", "Porto", "Kobe",
          "Basel", "Tampa", "Lyon", "Gdansk", "Nantes", "Sapporo"]
STOCK = PRODUCTS + ["tablet", "router", "printer", "drone", "camera"]


def long_state_tasks():
    """One fact in an inventory of 20, 80 or 160 lines: whether a warehouse holds a
    product. Half the questions name a pair the list holds, at a random depth; the
    others a pair it does not, of which there are always some (30 cities x 15
    products)."""
    rng = random.Random(8)
    q_text = "Does the {city} warehouse hold any {product}?"
    tasks = []
    for lines in (20, 80, 160):
        items = []
        for i in range(30):
            pairs = set()
            while len(pairs) < lines:
                pairs.add((rng.choice(CITIES), rng.choice(STOCK)))
            rows = [f"The {city} warehouse holds {rng.randint(2, 900)} units of {product}." for city, product in pairs]
            rng.shuffle(rows)
            if i % 2 == 0:
                city, product = rng.choice(sorted(pairs))
                held = True
            else:
                while True:
                    city, product = rng.choice(CITIES), rng.choice(STOCK)
                    if (city, product) not in pairs:
                        break
                held = False
            state = "Inventory report.\n" + "\n".join(rows)
            q = noul(q_text.format(city=city, product=product))
            items.append(Item(f"inventory-{lines}-{i}", state, {"held": q}, {"held": held}))
        tasks.append(Task(f"long_state_{lines:03d}", "long state", f"A fact looked up in an inventory of {lines} lines.", items))
    return tasks


# ---------------------------------------------------------------- logic

NONSENSE = ["blickets", "wugs", "daxes", "feps", "tovs", "zorbs", "glims", "pleks", "snarps", "vurns"]
PEOPLE = ["Ana", "Ben", "Cleo", "Dev", "Eva", "Finn", "Gus", "Hana"]


def logic_tasks():
    rng = random.Random(9)
    follows_q = noul("Does the conclusion follow necessarily from the premises?")
    syllogisms = []
    for i in range(40):
        a, b, c = rng.sample(NONSENSE, 3)
        name = rng.choice(PEOPLE)
        form = i % 4
        if form == 0:      # every A is B; x is A; so x is B
            premises, conclusion, valid = [f"Every {a[:-1]} is a {b[:-1]}.", f"{name} is a {a[:-1]}."], f"{name} is a {b[:-1]}.", True
        elif form == 1:    # every A is B; every B is C; so every A is C
            premises, conclusion, valid = [f"All {a} are {b}.", f"All {b} are {c}."], f"All {a} are {c}.", True
        elif form == 2:    # every A is B; x is B; so x is A (affirming the consequent)
            premises, conclusion, valid = [f"Every {a[:-1]} is a {b[:-1]}.", f"{name} is a {b[:-1]}."], f"{name} is a {a[:-1]}.", False
        else:              # some A are B; some B are C; so some A are C (undistributed middle)
            premises, conclusion, valid = [f"Some {a} are {b}.", f"Some {b} are {c}."], f"Some {a} are {c}.", False
        syllogisms.append(Item(f"syllogism-{i}", {"premises": premises, "conclusion": conclusion}, {"follows": follows_q}, {"follows": valid}))
    rng.shuffle(syllogisms)

    tallest = []
    for i in range(40):
        names = rng.sample(PEOPLE, 3 + i % 3)
        facts = [f"{names[k]} is taller than {names[k + 1]}." for k in range(len(names) - 1)]
        rng.shuffle(facts)
        options = sorted(names)
        q = choice("Who is the tallest?", {n: None for n in options})
        tallest.append(Item(f"tallest-{i}", " ".join(facts), {"tallest": q}, {"tallest": names[0]}))

    weather = ["rainy", "sunny", "snowy", "foggy", "windy"]
    negation = []
    for i in range(40):
        today = rng.choice(weather)
        asked = rng.choice(weather)
        negated = i % 2 == 1
        text = f"Is it not {asked} today?" if negated else f"Is it {asked} today?"
        truth = (today != asked) if negated else (today == asked)
        negation.append(Item(f"negation-{i}", {"forecast": f"Today will be {today}."}, {"answer": noul(text)}, {"answer": truth}))
    return [
        Task("logic_syllogism", "logic", "Whether a conclusion follows from two premises (two valid, two invalid forms).", syllogisms),
        Task("logic_order", "logic", "The tallest of three to five people from pairwise comparisons.", tallest),
        Task("logic_negation", "logic", "Yes or no questions about a forecast, half of them negated.", negation),
    ]


# ---------------------------------------------------------------- language identification and code

LANGUAGES = [
    ("English", "The weather is lovely today, let's go for a walk."), ("English", "Please send me the report by Friday."),
    ("English", "My brother lives near the station."),
    ("Spanish", "El tiempo está muy bonito hoy, vamos a pasear."), ("Spanish", "Por favor, envíame el informe antes del viernes."),
    ("Spanish", "Mi hermano vive cerca de la estación."),
    ("French", "Il fait très beau aujourd'hui, allons nous promener."), ("French", "Merci de m'envoyer le rapport avant vendredi."),
    ("French", "Mon frère habite près de la gare."),
    ("German", "Das Wetter ist heute schön, lass uns spazieren gehen."), ("German", "Bitte schick mir den Bericht bis Freitag."),
    ("German", "Mein Bruder wohnt in der Nähe des Bahnhofs."),
    ("Italian", "Oggi il tempo è bellissimo, andiamo a fare una passeggiata."), ("Italian", "Per favore, mandami il rapporto entro venerdì."),
    ("Italian", "Mio fratello abita vicino alla stazione."),
    ("Portuguese", "O tempo está ótimo hoje, vamos dar um passeio."), ("Portuguese", "Por favor, envie-me o relatório até sexta-feira."),
    ("Portuguese", "O meu irmão mora perto da estação."),
    ("Dutch", "Het weer is vandaag prachtig, laten we gaan wandelen."), ("Dutch", "Stuur me alsjeblieft het rapport voor vrijdag."),
    ("Dutch", "Mijn broer woont vlakbij het station."),
    ("Swedish", "Vädret är underbart idag, låt oss gå på en promenad."), ("Swedish", "Skicka mig rapporten senast på fredag."),
    ("Swedish", "Min bror bor nära stationen."),
    ("Polish", "Pogoda jest dziś piękna, chodźmy na spacer."), ("Polish", "Proszę, wyślij mi raport do piątku."),
    ("Polish", "Mój brat mieszka niedaleko dworca."),
    ("Turkish", "Bugün hava çok güzel, hadi yürüyüşe çıkalım."), ("Turkish", "Lütfen raporu cumaya kadar gönder."),
    ("Turkish", "Kardeşim istasyonun yakınında oturuyor."),
]

CODE = [
    ("python", "def add(a, b):\n    return a + b\n\nprint(add(2, 3))"),
    ("python", "items = [x * 2 for x in range(10) if x % 2 == 0]"),
    ("python", "with open('data.txt') as f:\n    lines = f.readlines()"),
    ("python", "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"),
    ("javascript", "function add(a, b) {\n  return a + b;\n}\nconsole.log(add(2, 3));"),
    ("javascript", "const items = [...Array(10).keys()].filter(x => x % 2 === 0);"),
    ("javascript", "document.querySelector('#btn').addEventListener('click', () => alert('hi'));"),
    ("javascript", "export default async function load() {\n  const res = await fetch('/api');\n  return res.json();\n}"),
    ("rust", "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"),
    ("rust", "let items: Vec<u32> = (0..10).filter(|x| x % 2 == 0).collect();"),
    ("rust", "impl Point {\n    fn new(x: f64, y: f64) -> Self { Point { x, y } }\n}"),
    ("rust", "match value {\n    Some(v) => println!(\"{v}\"),\n    None => {}\n}"),
    ("sql", "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 5;"),
    ("sql", "UPDATE users SET plan = 'pro' WHERE id = 42;"),
    ("sql", "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"),
    ("sql", "DELETE FROM sessions WHERE expires_at < NOW();"),
    ("go", "func add(a int, b int) int {\n\treturn a + b\n}"),
    ("go", "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hi\")\n}"),
    ("go", "for i := 0; i < 10; i++ {\n\tsum += i\n}"),
    ("go", "type Point struct {\n\tX, Y float64\n}"),
]


def code_snippet(rng, i):
    """A small Python snippet and whether running it raises an exception."""
    k = rng.randint(0, 5)
    templates = [
        f"values = [1, 2, 3]\nprint(values[{k}])",
        f"count = {k}\nprint(10 / count)",
        f"data = {{'a': 1, 'b': 2}}\nprint(data['{rng.choice('abc')}'])",
        f"text = '{rng.choice(['42', '7', 'seven', '3.5'])}'\nprint(int(text))",
        f"items = []\nfor i in range({k}):\n    items.append(i)\nprint(items[0])",
    ]
    code = templates[i % len(templates)]
    try:
        exec(compile(code, "<snippet>", "exec"), {"print": lambda *a: None})
        raises = False
    except Exception:
        raises = True
    return code, raises


def text_tasks():
    rng = random.Random(10)
    names = sorted({lang for lang, _ in LANGUAGES})
    lang_q = choice("Which language is the text written in?", {n: None for n in names})
    language_id = [Item(f"lang-{i}", text, {"language": lang_q}, {"language": lang}) for i, (lang, text) in enumerate(LANGUAGES)]
    rng.shuffle(language_id)
    code_names = ["go", "javascript", "python", "rust", "sql"]
    code_q = choice("Which programming language is this code written in?", {n: None for n in code_names})
    code_language = [Item(f"code-{i}", code, {"language": code_q}, {"language": lang}) for i, (lang, code) in enumerate(CODE)]
    rng.shuffle(code_language)
    raises_q = noul("Does running this Python code raise an exception?")

    drawn = iter(range(10**6))

    def snippet():
        code, raises = code_snippet(rng, rng.randint(0, 4))
        return Item(f"raises-{next(drawn)}", code, {"raises": raises_q}, {"raises": raises}), raises

    code_raises = balanced(rng, snippet, 20, 20)
    return [
        Task("language_id", "language", "The language of a sentence, of ten European languages and Turkish.", language_id),
        Task("code_language", "code", "The programming language of a snippet, of five.", code_language),
        Task("code_raises", "code", "Whether a short Python snippet raises an exception when run.", code_raises),
    ]


# ---------------------------------------------------------------- robustness

def reorder_options(item, order):
    questions = {}
    for qid, q in item.questions.items():
        q = dict(q)
        if q["type"] == "choice":
            keys = list(q["criteria"])
            keys = list(reversed(keys)) if order == "reversed" else keys[1:] + keys[:1]
            q["criteria"] = {k: q["criteria"][k] for k in keys}
        questions[qid] = q
    return questions


def reorder_keys(value, rng):
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        return {k: reorder_keys(value[k], rng) for k in keys}
    if isinstance(value, list):
        return [reorder_keys(v, rng) for v in value]
    return value


REWORDED = {
    "sentiment": "How does the writer feel, judging by this text?",
    "ticket_routing": "Who should take care of this customer message?",
    "capital": "Is the given city the seat of government of the given country?",
    "class": "What kind of animal is this?",
    "plan": "What subscription does this account have?",
    "continent": "Which continent contains this location?",
}


def robustness_tasks(base):
    """Rewrites of items from `base` (a name -> task map) that keep their answers."""
    rng = random.Random(11)
    out = []
    for name in ("sentiment", "ticket_routing", "animal_class", "geo_continent"):
        for order in ("reversed", "rotated"):
            items = [Item(f"{i.id}~{order}", i.state, reorder_options(i, order), i.gold, variant_of=i.id) for i in base[name].items]
            out.append(Task(f"{name}~options_{order}", "robustness", f"{name} with the options {order}.", items))
    for name in ("sentiment", "ticket_routing", "capitals", "animal_class", "json_lookup", "geo_continent"):
        items = []
        for i in base[name].items:
            questions = {}
            for qid, q in i.questions.items():
                q = dict(q)
                q["instructions"] = REWORDED.get(qid, REWORDED.get(name, q["instructions"]))
                questions[qid] = q
            items.append(Item(f"{i.id}~reworded", i.state, questions, i.gold, variant_of=i.id))
        out.append(Task(f"{name}~reworded", "robustness", f"{name} with its instructions reworded.", items))
    for name in ("json_lookup", "json_order_total"):
        items = [Item(f"{i.id}~keys", reorder_keys(i.state, rng), i.questions, i.gold, variant_of=i.id) for i in base[name].items]
        out.append(Task(f"{name}~keys_shuffled", "robustness", f"{name} with the state's keys in another order.", items))
    return out


def all_tasks(world):
    base = (geo_tasks(world) + math_tasks() + json_tasks() + time_tasks() + language_tasks() + knowledge_tasks()
            + option_scaling_tasks() + long_state_tasks() + logic_tasks() + text_tasks())
    return base + robustness_tasks({t.name: t for t in base})
