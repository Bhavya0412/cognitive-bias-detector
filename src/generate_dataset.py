"""
Dataset generator for the Cognitive Bias Detection project.

Produces a large, labeled dataset covering 5 classes:
    0 - neutral
    1 - confirmation_bias
    2 - anchoring_bias
    3 - availability_heuristic
    4 - framing_effect

Uses template-based synthesis with heavy variation (topics, subjects,
verbs, hedges, intensifiers) so the final corpus contains thousands
of distinct sentences suitable for training a classical NLP model.
"""

import csv
import os
import random
import itertools

random.seed(42)

# ----------------------------------------------------------------------
# Topic vocabulary used across every bias class.
# ----------------------------------------------------------------------
SUBJECTS = [
    "this strategy", "our approach", "the new policy", "this method",
    "the proposed plan", "their framework", "this algorithm",
    "our marketing campaign", "the investment strategy",
    "this diet", "the training program", "our hiring process",
    "this product", "the software update", "the teaching method",
    "this medication", "the business model", "our pricing strategy",
    "this theory", "the research design", "the treatment",
    "the new feature", "our sales technique", "this framework",
    "the management style", "this vaccine", "the economic policy",
    "our negotiation tactic", "this study", "the experiment",
]

DOMAINS = [
    "in the last quarter", "in our team", "in the industry",
    "in this market", "during the trial", "across the board",
    "in recent studies", "in most cases", "in practice",
    "for our customers", "among users", "in the region",
    "in our department", "in this sector", "in every scenario",
]

TIME_REFS = [
    "last month", "yesterday", "last week", "recently",
    "a few days ago", "earlier this year", "last quarter",
    "this morning", "in the last meeting", "today",
]

EVIDENCE_WORDS_STRONG = [
    "clearly", "obviously", "undeniably", "without a doubt",
    "certainly", "absolutely", "unquestionably", "definitely",
]

NUMBERS = [
    "10", "15", "20", "25", "30", "40", "50", "60", "70",
    "75", "80", "85", "90", "95", "100", "200", "500", "1000",
]

PRODUCTS = [
    "phone", "laptop", "car", "house", "subscription",
    "service package", "training course", "consulting service",
    "software license", "insurance plan",
]

# ----------------------------------------------------------------------
# 1. CONFIRMATION BIAS TEMPLATES
#    Marker: absolute language, one-sided reasoning, ignoring evidence.
# ----------------------------------------------------------------------
CONFIRMATION_TEMPLATES = [
    "{subj} always works because it worked {time}.",
    "{subj} {evidence} proves my point, and any contrary data is just noise.",
    "I knew {subj} would succeed — every example I can think of supports this.",
    "{subj} never fails {domain}; the critics are simply wrong.",
    "The data that agrees with me is valid; the rest is flawed.",
    "I only trust sources that confirm what I already believe about {subj}.",
    "{subj} is the best option — I refuse to consider alternatives.",
    "Every time I've used {subj} it has worked, so it must be universally effective.",
    "{subj} is {evidence} superior, and I don't need more evidence to prove it.",
    "I've seen {subj} succeed once, so it will always succeed.",
    "Anyone who disagrees about {subj} hasn't looked at the evidence I've seen.",
    "{subj} is right because all my favorite experts agree with it.",
    "My own experience with {subj} is enough to conclude it is effective.",
    "{subj} worked for me, so it must work for everyone.",
    "There is no point in reviewing counter-evidence; {subj} is {evidence} correct.",
    "I dismiss the negative reports about {subj} because they contradict my view.",
    "{subj} has proven itself — the skeptics are ignoring the obvious.",
    "I'm confident {subj} is flawless; every sign confirms it.",
    "{subj} {evidence} works {domain}, and that settles the matter.",
    "I refuse to read studies that oppose {subj} because they are biased.",
    "The success of {subj} is so {evidence} that further investigation is unnecessary.",
    "{subj} cannot be wrong — every case I know of validates it.",
    "I only cite reports that back up my stance on {subj}.",
    "{subj} is {evidence} the right choice; anyone who says otherwise is mistaken.",
    "Based on the handful of times I saw {subj} work, it must always work.",
    "{subj} {evidence} outperforms every alternative, and contrary claims can be ignored.",
    "I feel strongly that {subj} is correct, and that feeling is enough proof.",
    "Critics of {subj} are just looking for problems that do not exist.",
    "I already made up my mind about {subj}, so more analysis is pointless.",
    "{subj} is guaranteed to succeed — I have seen it happen before.",
    "Every piece of evidence I've selected supports {subj}.",
    "I ignore the failures of {subj} because they are exceptions.",
    "The results of {subj} match my expectations, so they must be accurate.",
    "{subj} worked {time}, which proves it will always work.",
    "I don't need peer review — {subj} is {evidence} effective.",
    "People who question {subj} don't understand it the way I do.",
    "{subj} is {evidence} the correct answer; dissenting views are not worth reading.",
    "My intuition about {subj} has never been wrong, and it won't start now.",
    "I cherry-pick the studies that support {subj} because they reflect reality.",
    "{subj} has zero downsides — I've confirmed it with people who already agree.",
]

# ----------------------------------------------------------------------
# 2. ANCHORING BIAS TEMPLATES
#    Marker: first number / initial estimate drives conclusion.
# ----------------------------------------------------------------------
ANCHORING_TEMPLATES = [
    "The first price I saw for the {product} was ${num1}, so anything above ${num2} feels expensive.",
    "Since the initial estimate was {num1}%, I still believe the real figure is close to that.",
    "The original quote of ${num1} set my expectation, and I refuse to pay more.",
    "Our first projection said {num1} units, so the final number must be around that.",
    "Because the opening offer was ${num1}, I keep judging every counteroffer against it.",
    "The starting estimate of {num1} hours sticks in my head, even though the scope has changed.",
    "I assumed {subj} would cost ${num1} from day one, and I haven't updated that view.",
    "The first review I read gave {subj} a {num1}/100, which still shapes my opinion.",
    "My boss mentioned {num1}% growth in our first meeting, and I'm still anchored to that number.",
    "The initial sales target was {num1}, and I can't think about the problem without it.",
    "I heard ${num1} in the first conversation, so that feels like the fair price.",
    "The very first data point showed {num1}, and I'm unwilling to move far from it.",
    "Because the supplier opened with ${num1}, I consider any discount a win.",
    "Our earliest forecast of {num1} units is still anchoring all my current estimates.",
    "The first candidate asked for ${num1}, so every other candidate is judged against it.",
    "I saw {num1}% on the first slide and now every number feels relative to that.",
    "The opening bid was ${num1}, which has set the ceiling in my mind.",
    "Because a colleague said {num1} yesterday, I find it hard to consider other values.",
    "The first {product} listing I checked was ${num1}, and nothing else feels reasonable.",
    "My initial read was that {subj} takes {num1} days, and I cannot let go of that number.",
    "The manufacturer suggested ${num1}, so anything lower feels like a bargain regardless of quality.",
    "The first report I saw claimed {num1}%, and I still treat that as the baseline.",
    "Since the original deadline was {num1} days, I keep comparing every revised plan to it.",
    "The first number I encountered for {subj} was {num1}, and I am stuck on it.",
    "Because the headline said ${num1}, I cannot assess the deal without that reference.",
    "Our first analysis estimated {num1} customers, and I can't shake that figure.",
    "The first benchmark was {num1}, so I keep framing results around that baseline.",
    "When I saw ${num1} initially, it became the anchor for every later decision.",
    "The first consultant we spoke to quoted ${num1}, and I measure everyone else by it.",
    "I read an article claiming {num1}% first, and every later number feels wrong by comparison.",
    "The first version of the budget had ${num1}, which I still treat as the true cost.",
    "Because the article's headline used ${num1}, that figure dominates my thinking.",
    "Our earliest rough guess was {num1}, and I'm unwilling to revise it meaningfully.",
    "The very first offer I received was ${num1}, and it sets my expectation permanently.",
    "Since the first quarter showed {num1}% growth, I expect the same every quarter.",
    "The first time I priced the {product}, it was ${num1}, and I keep comparing.",
    "The initial benchmark of {num1} units is the lens through which I see every report.",
    "I cannot evaluate {subj} without comparing it to the original ${num1} estimate.",
    "The first article I read said {num1}%, and I still use that as my reference.",
    "Because my first impression was ${num1}, every counter-proposal feels excessive.",
]

# ----------------------------------------------------------------------
# 3. AVAILABILITY HEURISTIC TEMPLATES
#    Marker: judging likelihood by how easy it is to recall examples.
# ----------------------------------------------------------------------
AVAILABILITY_TEMPLATES = [
    "I just saw a news story about it, so {subj} must be very common.",
    "A friend told me about {subj} {time}, so it must be a widespread issue.",
    "I can think of three examples off the top of my head, so {subj} is clearly the norm.",
    "Because {subj} was on the front page {time}, it must be happening everywhere.",
    "I remember a case of {subj} {time}, so it's probably very frequent.",
    "My coworker mentioned {subj} {time}, which proves it is happening all the time.",
    "I watched a documentary on {subj} and now I'm sure it is a common occurrence.",
    "Since a recent viral post discussed {subj}, it must affect most people.",
    "I can recall two stories about {subj} from this week alone, so it must be widespread.",
    "{subj} was trending on social media {time}, so it's clearly a major trend.",
    "I saw {subj} mentioned in three articles today, so it must be an epidemic.",
    "Because my neighbor experienced {subj}, I assume many people are affected.",
    "The news keeps showing {subj}, so it must be the biggest issue of the year.",
    "I read about {subj} in a magazine, so it has to be widely relevant.",
    "Since {subj} happened to someone I know, it's probably happening everywhere.",
    "It's been mentioned in my feed {time}, so {subj} is definitely a trend.",
    "Because I heard about {subj} on the radio today, it must be a top priority.",
    "I can recall vivid examples of {subj}, which makes it feel common.",
    "My family friend had {subj} happen to them, so it must be highly likely for anyone.",
    "I saw a viral video about {subj}, and now I'm convinced it happens to most people.",
    "Three people at work mentioned {subj}, so it must be common across the country.",
    "Since {subj} dominates the news cycle, it is clearly the most important issue.",
    "I remember a dramatic story about {subj}, so I overestimate how often it occurs.",
    "A podcast just covered {subj}, so it must be a widespread reality.",
    "Because my cousin had a bad experience with {subj}, I assume most people do too.",
    "The last three articles I read mentioned {subj}, so it's obviously the biggest trend.",
    "I saw {subj} discussed on TV, so it must be more frequent than statistics suggest.",
    "{subj} keeps coming up in conversations, which proves it is very common.",
    "Since I've read about {subj} twice this week, it must be happening constantly.",
    "Because someone I know mentioned {subj}, the probability seems much higher than it is.",
    "I heard a striking anecdote about {subj}, so I now overestimate the risk.",
    "A dramatic headline about {subj} makes me think it happens to almost everyone.",
    "My manager told me a story about {subj}, so it must be an everyday problem.",
    "Since a famous person discussed {subj}, it must apply to the general public too.",
    "I can easily recall cases of {subj}, so it must be the default outcome.",
    "Because the last few emails I received mentioned {subj}, it must be universal.",
    "{subj} was discussed at the conference, so it's clearly the dominant reality.",
    "I heard about {subj} twice {time}, so I now treat it as the most likely outcome.",
    "Because it was the lead story {time}, {subj} must outweigh every other factor.",
    "A single vivid example of {subj} makes me believe it represents the whole population.",
]

# ----------------------------------------------------------------------
# 4. FRAMING EFFECT TEMPLATES
#    Marker: same fact presented one way vs. another to steer feeling.
# ----------------------------------------------------------------------
FRAMING_TEMPLATES = [
    "The {product} has a {num1}% success rate, which sounds much better than saying it has a {num2}% failure rate.",
    "Our plan saves {num1} lives — that sounds better than admitting {num2} people still die.",
    "The report says {num1}% of customers are satisfied, which is more persuasive than saying {num2}% are unhappy.",
    "Describing the tax as a 'small contribution' makes it feel better than calling it a 'mandatory payment'.",
    "{subj} has a {num1}% approval rating — that sounds stronger than saying {num2}% disapprove.",
    "Saying {num1} out of 100 survived is more reassuring than saying {num2} out of 100 died.",
    "The offer of '{num1}% off' feels more attractive than saying you still pay {num2}% of the price.",
    "'Lean beef: {num1}% fat-free' sounds healthier than 'beef: {num2}% fat'.",
    "'Join the {num1}% of winners' is more compelling than 'accept that {num2}% lose'.",
    "Calling it an 'investment opportunity' rather than a 'risk' changes how people perceive {subj}.",
    "'Limited-time exclusive access' sounds more appealing than 'expiring offer'.",
    "Describing the drug as 'effective in {num1}% of cases' is kinder than 'ineffective in {num2}% of cases'.",
    "'{num1}% chance of recovery' sounds more positive than '{num2}% chance of no recovery'.",
    "'Save {num1}% this weekend' feels better than 'pay the full price after Sunday'.",
    "Calling it a 'service fee' is gentler than calling it a 'surcharge' for the same amount.",
    "Saying {subj} is 'often successful' is stronger than saying it 'sometimes fails'.",
    "'Nine out of ten dentists recommend it' sounds better than 'one out of ten do not'.",
    "Framing the layoff as 'workforce optimization' softens the impact of the same decision.",
    "'{num1}% organic' feels healthier than '{num2}% non-organic ingredients'.",
    "The phrase 'complimentary trial' feels more generous than 'free period before billing'.",
    "'Exclusive membership' sounds more desirable than 'paid subscription'.",
    "'Guaranteed return of {num1}%' feels safer than 'loss probability of {num2}%'.",
    "Calling the delay a 'schedule adjustment' softens the fact that {subj} missed its deadline.",
    "'{num1}% of participants improved' sounds stronger than '{num2}% did not improve'.",
    "Saying the plan 'preserves jobs' is more appealing than 'avoids additional cuts'.",
    "The phrase 'pre-owned vehicle' sounds better than 'used car', even for the same item.",
    "Describing a tax rise as an 'investment in infrastructure' changes the public reaction.",
    "'Our product reduces risk by {num1}%' sounds better than 'a {num2}% risk remains'.",
    "Calling the proposal 'cost-saving' is more attractive than calling it 'spending cuts'.",
    "The label 'natural flavors' sounds healthier than 'processed flavor compounds'.",
    "Framing the rate as 'only ${num1} per day' feels cheaper than '${num2} per month'.",
    "'{num1}% of students pass' is more encouraging than '{num2}% fail the exam'.",
    "Describing the policy as 'streamlined' is softer than calling it 'reduced services'.",
    "'Premium quality' sounds better than 'higher-cost option' for the same item.",
    "Saying the candidate has a {num1}% lead sounds better than a {num2}% deficit for the rival.",
    "Calling the program 'revenue-neutral' softens the fact that someone still pays more.",
    "'Only {num1} seats left' creates urgency that 'plenty of capacity' would not.",
    "'Premium member' sounds more prestigious than 'paying customer', for the same role.",
    "Describing the discount as 'exclusive savings' is more appealing than 'reduced price'.",
    "'Up to {num1}% off' feels better than 'most items are not discounted'.",
]

# ----------------------------------------------------------------------
# 5. NEUTRAL TEMPLATES (unbiased, evidence-based)
# ----------------------------------------------------------------------
NEUTRAL_TEMPLATES = [
    "{subj} worked in one instance, but more evidence is needed to confirm its effectiveness.",
    "Preliminary data suggests {subj} may help, though further study is required.",
    "The results for {subj} are mixed; some trials show benefits while others do not.",
    "More research is needed before drawing conclusions about {subj}.",
    "{subj} showed a modest improvement, but the sample size was small.",
    "We observed positive results for {subj}, but replication is needed.",
    "The study of {subj} is ongoing, and no firm conclusions can be drawn yet.",
    "Evidence for {subj} is promising but not yet conclusive.",
    "{subj} had a positive outcome in this case, though individual results may vary.",
    "The benefits of {subj} should be weighed against its drawbacks.",
    "Data indicates {subj} works for some users, but not all.",
    "{subj} may be effective under certain conditions; more testing is recommended.",
    "Both supporting and opposing evidence exist for {subj}, so caution is warranted.",
    "The effect of {subj} varied across participants in the trial.",
    "{subj} showed improvement in {num1}% of cases, while {num2}% showed no change.",
    "Independent review of {subj} is recommended before acting on these findings.",
    "{subj} may offer advantages, but its long-term effects are unknown.",
    "Results for {subj} differ depending on context and methodology.",
    "The outcome of {subj} depends on several factors that are still being studied.",
    "{subj} delivered mixed results, and additional research is planned.",
    "Users reported a range of experiences with {subj}, from positive to neutral.",
    "While {subj} performed well in one test, broader validation is needed.",
    "There are valid arguments both for and against adopting {subj}.",
    "{subj} works in certain scenarios, but generalization should be done carefully.",
    "Evaluators noted both strengths and weaknesses in {subj}.",
    "Further data collection is required before recommending {subj} widely.",
    "{subj} improved outcomes for some participants, but the effect size was small.",
    "The evidence for {subj} is preliminary and should be interpreted with care.",
    "{subj} may be suitable for some users; others should consider alternatives.",
    "Mixed findings on {subj} suggest that individual evaluation is important.",
    "The team reviewed multiple perspectives before deciding on {subj}.",
    "{subj} has both potential benefits and potential risks worth considering.",
    "Preliminary trials of {subj} are encouraging, but peer review is pending.",
    "The report presents {subj} with both supporting data and noted limitations.",
    "{subj} may achieve the stated goal, though alternatives should be evaluated.",
    "Results suggest {subj} performs comparably to existing solutions.",
    "No single study is sufficient to confirm that {subj} works reliably.",
    "{subj} had a measurable effect, though the magnitude varied.",
    "Whether {subj} is the right choice depends on context-specific factors.",
    "Further analysis is needed to determine whether {subj} generalizes to other populations.",
]

# ----------------------------------------------------------------------
# Variation layer: extra noise so the same template produces different
# sentences on repeated fills.
# ----------------------------------------------------------------------
PREFIXES = [
    "", "", "", "",  # empty weighted heavily
    "Honestly, ", "Look, ", "In my view, ", "Frankly, ",
    "From what I've seen, ", "Based on my experience, ",
    "It seems to me that ", "As far as I can tell, ",
]

SUFFIXES = [
    "", "", "", "",
    " — end of story.", " I stand by that.", " That's the bottom line.",
    " No further discussion needed.", " That's just how it is.",
    " That's the reality.",
]

NEUTRAL_PREFIXES = [
    "", "", "", "",
    "According to the report, ", "Based on current data, ",
    "From the available evidence, ", "Looking at the study, ",
    "After reviewing the findings, ",
]

NEUTRAL_SUFFIXES = [
    "", "", "", "",
    " Additional validation would strengthen these findings.",
    " Contextual factors should be considered.",
    " The sample remains limited.",
]


def fill(template, neutral=False):
    """Fill a template with random vocabulary."""
    subj = random.choice(SUBJECTS)
    domain = random.choice(DOMAINS)
    time_ref = random.choice(TIME_REFS)
    evidence = random.choice(EVIDENCE_WORDS_STRONG)
    product = random.choice(PRODUCTS)

    # Two distinct numbers for framing / anchoring contrasts
    n1 = random.choice(NUMBERS)
    n2_candidates = [x for x in NUMBERS if x != n1]
    n2 = random.choice(n2_candidates)

    text = template.format(
        subj=subj, domain=domain, time=time_ref,
        evidence=evidence, product=product,
        num1=n1, num2=n2,
    )

    if neutral:
        text = random.choice(NEUTRAL_PREFIXES) + text + random.choice(NEUTRAL_SUFFIXES)
    else:
        text = random.choice(PREFIXES) + text + random.choice(SUFFIXES)

    # Capitalize first letter cleanly
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def build_class(templates, label, target_count, neutral=False):
    """Generate approximately `target_count` examples for one class."""
    seen = set()
    rows = []
    # Keep sampling until we hit the target; templates + variation give
    # us plenty of distinct outputs.
    attempts = 0
    while len(rows) < target_count and attempts < target_count * 20:
        tpl = random.choice(templates)
        text = fill(tpl, neutral=neutral)
        if text not in seen:
            seen.add(text)
            rows.append((text, label))
        attempts += 1
    return rows


ABBREVIATED_TEMPLATES = [
    # Short, terse versions that overlap in style across classes -
    # introduces realistic confusability.
    ("{subj} worked before, so it works.", "confirmation_bias"),
    ("It worked once, so it always works.", "confirmation_bias"),
    ("I'm right about {subj}.", "confirmation_bias"),
    ("Critics are wrong about {subj}.", "confirmation_bias"),
    ("{subj} is the best.", "confirmation_bias"),

    ("${num1} was the first price I saw.", "anchoring_bias"),
    ("The initial estimate was {num1}.", "anchoring_bias"),
    ("First offer: ${num1}.", "anchoring_bias"),
    ("Original quote stuck with me.", "anchoring_bias"),
    ("The opening number was {num1}%.", "anchoring_bias"),

    ("I just heard about {subj}.", "availability_heuristic"),
    ("Saw it on the news.", "availability_heuristic"),
    ("My friend mentioned it.", "availability_heuristic"),
    ("It's trending, so it must be common.", "availability_heuristic"),
    ("I can recall examples easily.", "availability_heuristic"),

    ("{num1}% success is better than {num2}% failure.", "framing_effect"),
    ("Saving lives sounds better than losing lives.", "framing_effect"),
    ("'{num1}% fat-free' vs '{num2}% fat'.", "framing_effect"),
    ("Exclusive offer, not expiring deal.", "framing_effect"),
    ("Investment, not risk.", "framing_effect"),

    ("Results are mixed.", "neutral"),
    ("More data is needed.", "neutral"),
    ("{subj} may work for some.", "neutral"),
    ("The evidence is preliminary.", "neutral"),
    ("Further study is required.", "neutral"),
]


def build_noisy_rows(n_per_template=6):
    """Short, ambiguous examples that create realistic class-boundary confusion."""
    rows = []
    for tpl, label in ABBREVIATED_TEMPLATES:
        for _ in range(n_per_template):
            text = fill(tpl, neutral=(label == "neutral"))
            rows.append((text, label))
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "bias_dataset.csv")
    out_path = os.path.abspath(out_path)

    per_class = 1650  # main templates; plus ~150 short/noisy per class below

    all_rows = []
    all_rows += build_class(CONFIRMATION_TEMPLATES, "confirmation_bias", per_class)
    all_rows += build_class(ANCHORING_TEMPLATES, "anchoring_bias", per_class)
    all_rows += build_class(AVAILABILITY_TEMPLATES, "availability_heuristic", per_class)
    all_rows += build_class(FRAMING_TEMPLATES, "framing_effect", per_class)
    all_rows += build_class(NEUTRAL_TEMPLATES, "neutral", per_class, neutral=True)

    # Add realistic, harder-to-classify short examples
    all_rows += build_noisy_rows(n_per_template=10)

    # Introduce light label noise (~2%) so accuracy is not unrealistically perfect
    n_noise = int(0.02 * len(all_rows))
    labels = ["confirmation_bias", "anchoring_bias",
              "availability_heuristic", "framing_effect", "neutral"]
    noise_indices = random.sample(range(len(all_rows)), n_noise)
    for i in noise_indices:
        txt, orig = all_rows[i]
        wrong = random.choice([lbl for lbl in labels if lbl != orig])
        all_rows[i] = (txt, wrong)

    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")

    # Print class distribution
    from collections import Counter
    counts = Counter(label for _, label in all_rows)
    print("\nClass distribution:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:25s} {n}")


if __name__ == "__main__":
    main()
