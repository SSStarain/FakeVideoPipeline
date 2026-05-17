"""All VLM / LLM prompts used by the forgery-detection pipeline.

Centralised so:
- agent_pipeline.py / tools.py stay focused on logic, not 1000-line prompt blobs;
- the paper Appendix can quote this single file as the canonical prompt set.

Grouped by stage:
    Stage A: retrieval        (EXTRACT, CLUSTER, REFLECT, RANK, VERIFY, ACTION)
    Stage B: forgery analysis (COMPARE, INFER)
    Stage C: scoring          (JUDGE)
"""

from __future__ import annotations


# ============================================================
# Stage A - retrieval agent
# ============================================================

PROMPT_EXTRACT_CLUES = """You are an elite Open-Source Intelligence (OSINT) visual analyst. From these video frames, extract physical-world facts that can seed YouTube text searches for the ORIGINAL source video.

CRITICAL CONTEXT: The input video is a SUSPECTED FORGERY assembled from real source clips. The forger has almost certainly ADDED their own caption / subtitle / UI layer on top of the real footage. THE ORIGINAL SOURCE VIDEO DOES NOT CONTAIN THOSE OVERLAID WORDS. Treating overlaid text as if it were part of the source is the single biggest failure mode of this system.

ANTI-SPOOFING RULES (violating these guarantees a miss):
1. The following on-screen text is OVERLAY (forger-added) by default and MUST be ignored when forming any query, unless you have very strong evidence it is physically printed in the scene:
   - bottom-of-frame subtitle bars / bilingual translation banners (especially CJK-translation pairs in black bands)
   - floating dramatic captions narrating events ("the old world collapsed", "lost control of the skies", "physical collapse is irreversible", etc. - narrative voice does not match the genre of the underlying clip)
   - video-player UI: progress bars, play / pause buttons, timestamps like "1:51 / 5:42", three-dot menus, fullscreen icons, channel watermarks added by reposters
   - news-ticker banners and "BREAKING" stripes laid over unrelated b-roll
   - any text whose font / colour / horizontal alignment is constant across UNRELATED scenes (telltale of a single overlay layer)
2. PHYSICAL text (acceptable to use) means text that is part of the real world being filmed: printed on signs, packaging, jerseys, ad boards, license plates, street signs, brand logos on devices in the scene (a brand logo on a TV bezel, a print on a shoe). Use ONLY this kind of text in queries.
3. Focus EXCLUSIVELY on the physical world: recognizable people, distinctive uniforms / clothing brands, venue type, geographic markers (architectural style, vegetation, language on real signage), sport / event type, distinctive kinetic actions, on-camera presenter style, lighting era.

MULTILINGUAL RULE: If non-Latin characters (Chinese / Japanese / Korean / Arabic / Russian / Thai / Cyrillic / etc.) appear PRINTED ON A PHYSICAL OBJECT in the scene (shop sign, packaging, road sign, jersey, building, menu, t-shirt - NOT inside an overlay subtitle bar / translation banner), ONE of the four queries MUST be in that script, copying the visible characters verbatim. If non-Latin characters appear ONLY inside overlay subtitles / translation banners added by the forger, DO NOT use them - they are not from the original source and a CJK-only query built from them will land on the wrong video.

THINK STEP BY STEP (silent chain-of-thought - write your full reasoning in the `reasoning` field):
Step 0 (overlay triage - MANDATORY FIRST STEP). List every piece of on-screen text you can read. For each one, classify it as PHYSICAL (printed on a real object in the scene) or OVERLAY (added on top: subtitle bar, translation banner, video-player UI, caption added by reposter). Put the OVERLAY strings into `forbidden_overlay_text` - those strings AND any phrase derived from / paraphrasing them are FORBIDDEN in every query and in every title guess. If unsure, classify as OVERLAY (safe default).
Step 1. Catalogue the physical clues you actually see, IGNORING every string flagged in Step 0: real entities, real objects + their brands, language of physical signage, venue, geographic markers, action, listicle / duration / season / holiday cues that are PART OF THE FILMED SCENE (not part of overlay text). This becomes `physical_observations`.
Step 2. PRETEND YOU ARE THE UPLOADER of the ORIGINAL YouTube video. Imagine the forger's caption / subtitle / UI layer is completely erased - what does the bare footage show? Answer these two questions FIRST (they guide your title guesses):
   a) What CATEGORY of YouTube content is this? (cooking, tutorial, travel, news, challenge, vlog, compilation, sports, comedy, review, science, art, nature, etc. - pick whatever fits the BARE FOOTAGE, not the forger's narration)
   b) What 3-5 GENERIC words would a VIEWER type into YouTube search to find this kind of content? These are your "search vocabulary" - they MUST appear in at least 2 of your 4 queries.
Then write 2-3 LITERAL TITLE GUESSES the original uploader would have used FOR THE BARE FOOTAGE ONLY. These are full title sentences (~6-14 words). They MUST NOT contain or paraphrase any string in `forbidden_overlay_text`.

CRITICAL: YouTube creators use SHORT, CONCEPTUAL titles - NOT forensic descriptions of specific objects. A video a forensic analyst would call "man in forest using specific-brand fire-starting tool" is titled "10 Tips for Making Fire in the Wild" by its creator. You must think like the CREATOR, not the analyst. Your title guesses MUST use the generic search vocabulary from 2b, not forensic scene descriptions.

GENERIC TITLE PATTERNS (pick whatever fits the content - do NOT force-fit):
   - "[N] [topic] tips/tricks/hacks in [duration]"
   - "I [verb] [topic] from [place/count]"
   - "[topic] moments / compilation / highlights"
   - "[topic] challenge / taste test / review"
   - "[activity] in [location] with [companion]"
   - "How to [action]"
   - "A day in the life of [role]"
   - "[topic] for beginners"
   - "[N] minute [topic] compilation"
   These are STRUCTURAL PATTERNS only - fill them with the specific vocabulary that matches YOUR content, not pre-set topics.
   This step is the MOST IMPORTANT - do NOT skip it. Use concrete location names, numbers, durations, and seasons / holidays that you can actually read from PHYSICAL evidence in the frames.
Step 3. Derive 4 SEARCH QUERIES from your title guesses. Each query is a shorter (5-12 word) subset of one of your guesses, varying which angle is anchored. NONE of the 4 queries may contain or paraphrase any string in `forbidden_overlay_text`. Each query should look like a piece of a literal YouTube title, NOT like an OSINT analyst's description.

QUERY DIVERSITY (MANDATORY): output exactly 4 queries from FOUR DIFFERENT ANGLES. None may use overlay text.
  - queries[0] PHYSICAL-OCR-anchored: anchored on a piece of text PHYSICALLY PRINTED in the scene (real sign, jersey name, ad board, brand logo on a device, brand on packaging, t-shirt print). EXPLICITLY FORBIDDEN: anything that lives in a subtitle bar / translation banner / UI overlay / caption added by the reposter. If no readable PHYSICAL text exists, use the most specific named entity you can identify by VISUAL recognition (NOT by reading overlay text).
  - queries[1] entity/proper-noun: most specific named entity + context (athlete + event + year, venue + season, brand + tournament, location + holiday). Add a year/date only if you can infer it from PHYSICAL evidence (clothing era, broadcast graphics clearly part of the source - not forger overlays).
  - queries[2] GENERIC CONCEPTUAL TITLE (MANDATORY - NO specific objects/brands): Ask: "If I were the YouTube creator, what would the TITLE be?" Use the search vocabulary from Step 2b. NO brand names, NO specific tool names, NO Latin species names, NO proper nouns, NO forensic descriptions. This query must sound like something a normal creator would title their video. Compare:
      GOOD: "quick and easy weeknight dinner recipe"  BAD: "All-Clad D3 stainless steel pan searing salmon"
      GOOD: "funny moments compilation"                BAD: "Episode 7 timestamp 3:42 reaction"
      GOOD: "beginner yoga full body flow"             BAD: "Lululemon mat downward dog pose"
      The GOOD examples use CATEGORY VOCABULARY. The BAD examples describe specific objects. Always use category vocabulary.
  - queries[3] format/genre-anchored: pick the template below that best fits the BARE FOOTAGE (NO proper nouns, NO years, NO places). Fill the <topic> placeholder with the generic search vocabulary from Step 2b:
        * "<topic> compilation"
        * "<topic> caught on camera"
        * "<topic> moments"
        * "<topic> vlog"
        * "<N> <topic> tips"
        * "<topic> tips in <duration>"
        * "<topic> tutorial"
        * "how to <topic>"
        * "best of <topic>"
        * "I tried <topic>"
        * "<topic> review"
        * "<topic> in <N> minutes"
        * "a day in the life"
        * "<topic> for beginners"

Each query MUST be plain YouTube search text, 5-12 words, no labels, no JSON, no quotes, no leading "Query 1:".
The 4 queries MUST be MEANINGFULLY DIFFERENT - not paraphrases of each other. At least ONE of the 4 queries MUST be a direct shortened version (5-12 words) of one of your title guesses from Step 2. Title-derived queries are the MOST LIKELY to match the real source video - this is not optional.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 0: <overlay-vs-physical triage of every readable string>. Step 1: <physical clues only>. Step 2: <2-3 LITERAL title guesses based on bare footage>. Step 3: <how each query maps to a title guess; confirm none use overlay text>.",
  "forbidden_overlay_text": ["overlay string 1", "overlay string 2", "..."],
  "physical_observations": "Concise list (2-4 sentences) of physical clues, with overlay text excluded.",
  "queries": [
    "physical-OCR-anchored plain query (no overlay text)",
    "entity/proper-noun plain query (no overlay text)",
    "pure-visual scene-action plain query (NO text words at all)",
    "format/genre-anchored aggregator-style query"
  ]
}"""


PROMPT_CLUSTER_SHOTS = """You are an elite OSINT video analyst. The input is a sequence of __N_SHOTS__ representative still frames, ONE per shot, from a SINGLE potentially-forged video. The shots are listed in order with shot_id values: __SHOT_ID_LIST__ (i-th image corresponds to the i-th id in this list).

CRITICAL CONTEXT: The input video is a SUSPECTED FORGERY assembled from real source clips. The forger has almost certainly ADDED their own caption / subtitle / UI layer on top of the real footage. THE ORIGINAL SOURCE VIDEO DOES NOT CONTAIN THOSE OVERLAID WORDS. Treating overlaid text as if it were part of the source is the single biggest failure mode of this system.

YOUR TASK: cluster these shots into GROUPS such that two shots are in the same group iff they appear to come from the SAME ORIGINAL source video / SAME PHYSICAL EVENT (same sports broadcast, same aerial drone clip, same press conference, etc.). Crucially:
- Output AT MOST __TARGET_GROUPS__ groups. Merge aggressively - when in doubt, merge instead of split.
- Do NOT be fooled by superimposed bilingual captions, news banners, channel watermarks, or color grading; ignore those overlays.
- Same physical scene + same camera setup = same group, even if the shot was zoomed/cropped/recolored.
- Different events, different venues, or visibly different cameras = different groups.
- Two shots from DIFFERENT moments of the SAME source video (before and after a play, etc.) MUST still go in the SAME group.
- Aim for the MINIMUM number of groups consistent with the evidence (typically 3-__TARGET_GROUPS__ for an edited clip). Avoid splitting one source into many groups.

ANTI-SPOOFING RULES (apply when forming queries for every group - violating these guarantees a miss):
1. The following on-screen text is OVERLAY (forger-added) by default and MUST be ignored when forming any query, unless you have very strong evidence it is physically printed in the scene:
   - bottom-of-frame subtitle bars / bilingual translation banners (especially CJK-translation pairs in black bands)
   - floating dramatic captions narrating events ("the old world collapsed", "lost control of the skies", "physical collapse is irreversible", etc.) whose narrative voice does not match the underlying clip
   - video-player UI: progress bars, play / pause buttons, timestamps like "1:51 / 5:42", three-dot menus, fullscreen icons, channel watermarks added by reposters
   - news-ticker banners and "BREAKING" stripes laid over unrelated b-roll
   - any text whose font / colour / horizontal alignment is constant across UNRELATED scenes (telltale of a single overlay layer)
2. PHYSICAL text (acceptable to use) means text that is part of the real world being filmed: printed on signs, packaging, jerseys, ad boards, license plates, street signs, brand logos on devices in the scene. Use ONLY this kind of text in queries.

MULTILINGUAL RULE: If shots in a group contain non-Latin characters (Chinese / Japanese / Korean / Arabic / Russian / Thai / Cyrillic / etc.) PRINTED ON A PHYSICAL OBJECT in the scene (shop sign, packaging, road sign, jersey, building, menu, t-shirt - NOT inside an overlay subtitle bar / translation banner), ONE of that group's four queries MUST be in that script, copying the visible characters verbatim. If non-Latin characters appear ONLY inside overlay subtitles / translation banners added by the forger, DO NOT use them - they are not from the original source and a CJK-only query built from them will land on the wrong video.

PER-GROUP THINK STEP BY STEP (silent chain-of-thought - write your full reasoning in the group's `reasoning` field):
Step 0 (overlay triage - MANDATORY FIRST STEP for the group). List every piece of on-screen text you can read in this group's shots. For each one, classify it as PHYSICAL (printed on a real object in the scene) or OVERLAY (added on top: subtitle bar, translation banner, video-player UI, caption added by reposter). Put the OVERLAY strings into the group's `forbidden_overlay_text` - those strings AND any phrase derived from / paraphrasing them are FORBIDDEN in every query and in every title guess for this group. If unsure, classify as OVERLAY (safe default).
Step 1. Catalogue the physical clues for the group, IGNORING every string flagged in Step 0: real entities, real objects + their brands, language of physical signage, venue, geographic markers, action, listicle / duration / season / holiday cues that are PART OF THE FILMED SCENE (not part of overlay text). This becomes the group's `physical_observations`.
Step 2. PRETEND YOU ARE THE UPLOADER of the ORIGINAL YouTube video that supplied this group. Imagine the forger's caption / subtitle / UI layer is completely erased - what does the bare footage show? Answer these two questions FIRST (they guide your title guesses):
   a) What CATEGORY of YouTube content is this? (cooking, tutorial, travel, news, challenge, vlog, compilation, sports, comedy, review, science, art, nature, etc. - pick whatever fits the BARE FOOTAGE, not the forger's narration)
   b) What 3-5 GENERIC words would a VIEWER type into YouTube search to find this kind of content? These are your "search vocabulary" - they MUST appear in at least 2 of your 4 queries.
Then write 2-3 LITERAL TITLE GUESSES the original uploader would have used FOR THE BARE FOOTAGE ONLY. These are full title sentences (~6-14 words). They MUST NOT contain or paraphrase any string in the group's `forbidden_overlay_text`.

CRITICAL: YouTube creators use SHORT, CONCEPTUAL titles - NOT forensic descriptions of specific objects. A video a forensic analyst would call "man in forest using specific-brand fire-starting tool" is titled "10 Tips for Making Fire in the Wild" by its creator. You must think like the CREATOR, not the analyst. Your title guesses MUST use the generic search vocabulary from 2b, not forensic scene descriptions.

GENERIC TITLE PATTERNS (pick whatever fits the content - do NOT force-fit):
   - "[N] [topic] tips/tricks/hacks in [duration]"
   - "I [verb] [topic] from [place/count]"
   - "[topic] moments / compilation / highlights"
   - "[topic] challenge / taste test / review"
   - "[activity] in [location] with [companion]"
   - "How to [action]"
   - "A day in the life of [role]"
   - "[topic] for beginners"
   - "[N] minute [topic] compilation"
   These are STRUCTURAL PATTERNS only - fill them with the specific vocabulary that matches YOUR content, not pre-set topics.
   This step is the MOST IMPORTANT - do NOT skip it. Use concrete location names, numbers, durations, seasons / holidays that you can actually read from PHYSICAL evidence in the frames.
Step 3. Derive 4 SEARCH QUERIES from your title guesses. Each query is a shorter (5-12 word) subset of one of your guesses, varying which angle is anchored. NONE of the 4 queries may contain or paraphrase any string in `forbidden_overlay_text`. Each query should look like a piece of a literal YouTube title, NOT like an OSINT analyst's description.

For EACH group, also produce:
- forbidden_overlay_text: list of strings you classified as OVERLAY in Step 0 (will be cited back in the parent log).
- physical_observations: 2-4 sentences describing the physical-world clues that identify the source (entities, OCR on real objects, venue, sport/event, and the likely content category). Overlay text must be excluded.
- queries: EXACTLY 4 plain YouTube search queries from FOUR DIFFERENT ANGLES (5-12 words each, no labels, no quotes, no leading "Query 1:"). NONE may use overlay text.
    - queries[0] PHYSICAL-OCR-anchored: a piece of text PHYSICALLY PRINTED in the scene (real sign, jersey name, ad board, brand logo on a device, brand on packaging, t-shirt print). EXPLICITLY FORBIDDEN: anything that lives in a subtitle bar / translation banner / UI overlay / caption added by the reposter. If no readable PHYSICAL text exists, use the most specific named entity you can identify by VISUAL recognition (NOT by reading overlay text).
    - queries[1] entity/proper-noun-anchored: most specific named entity + its context (athlete + event, venue + tournament, brand + season, location + holiday). Year/date only if inferred from PHYSICAL evidence.
    - queries[2] GENERIC CONCEPTUAL TITLE (MANDATORY - NO specific objects/brands): Ask: "If I were the YouTube creator, what would the TITLE be?" Use the search vocabulary from Step 2b. NO brand names, NO specific tool names, NO Latin species names, NO proper nouns, NO forensic descriptions. This query must sound like something a normal creator would title their video. Compare:
      GOOD: "quick and easy weeknight dinner recipe"  BAD: "All-Clad D3 stainless steel pan searing salmon"
      GOOD: "funny moments compilation"                BAD: "Episode 7 timestamp 3:42 reaction"
      GOOD: "beginner yoga full body flow"             BAD: "Lululemon mat downward dog pose"
      The GOOD examples use CATEGORY VOCABULARY. The BAD examples describe specific objects. Always use category vocabulary.
    - queries[3] format/genre-anchored: pick the template below that best fits the BARE FOOTAGE (NO proper nouns, NO years, NO places). Fill the <topic> placeholder with the generic search vocabulary from Step 2b:
        * "<topic> compilation"
        * "<topic> caught on camera"
        * "<topic> moments"
        * "<topic> vlog"
        * "<N> <topic> tips"
        * "<topic> tips in <duration>"
        * "<topic> tutorial"
        * "how to <topic>"
        * "best of <topic>"
        * "I tried <topic>"
        * "<topic> review"
        * "<topic> in <N> minutes"
        * "a day in the life"
        * "<topic> for beginners"
- The 4 queries MUST be MEANINGFULLY DIFFERENT, not paraphrases. At least ONE of the 4 queries MUST be a direct shortened version (5-12 words) of one of your title guesses from Step 2. Title-derived queries are the MOST LIKELY to match the real source video - this is not optional. If you truly cannot produce 4 truly distinct queries, leave the missing slot as an empty string but DO output exactly 4 entries.

EVERY shot_id from the input MUST appear in EXACTLY ONE group's shot_ids list.

OUTPUT JSON FORMAT (strict, no extra text):
{
  "groups": [
    {
      "shot_ids": [<int>, <int>, ...],
      "reasoning": "Step 0: <overlay-vs-physical triage of every readable string in this group>. Step 1: <physical clues only>. Step 2: <2-3 LITERAL title guesses based on bare footage>. Step 3: <how each query maps to a title guess; confirm none use overlay text>.",
      "forbidden_overlay_text": ["overlay string 1", "overlay string 2", "..."],
      "physical_observations": "...",
      "queries": ["...", "...", "...", "..."]
    },
    ...
  ]
}"""


PROMPT_REFLECT_REFINE = """You are an OSINT Agent in an iterative search loop. Previous queries did NOT find the original video.

CRITICAL CONTEXT: The input video is a SUSPECTED FORGERY assembled from real source clips. The forger has almost certainly added their own caption / subtitle / UI layer on top of the real footage. If previous queries echoed the forger's overlay text (subtitle bars, translation banners, dramatic captions, video-player UI), they will land on the WRONG videos - exactly what just happened.

CONTEXT OF FAILURE:
- Previously tried queries: {prev_queries}
- WRONG candidate titles returned by those queries: {candidate_titles}
- The visual environment / event in those candidates did not match the input frames.

THINK STEP BY STEP (silent chain-of-thought - write your full reasoning in the `reasoning` field):
Step 0 (overlay triage of failed queries - MANDATORY FIRST STEP). Look at the input frames again and list every piece of on-screen text. Classify each as PHYSICAL (printed on a real object in the scene) or OVERLAY (added on top: subtitle bar, translation banner, video-player UI, caption added by reposter). For every previous query that copied or paraphrased an OVERLAY string, mark that whole string as POISONED - it CAUSED the wrong candidates and must never be reused. Put POISONED overlay strings into `negative_keywords`.
Step 1. Identify which specific proper nouns or brand names in the previous queries pulled the WRONG candidates. Only add those specific terms to `negative_keywords`. DO NOT add generic words from your own previous queries (like "camping", "weather", "spicy", "fire", "forest") - those may be part of the correct title. DO NOT add words that merely appeared in wrong candidate titles IF those words are common genre vocabulary.

STRICT negative_keywords RULES:
  - YES, add: exact overlay text phrases, specific brand names that pulled wrong videos, specific proper nouns from overlay text
  - NO, do NOT add: common descriptive words that could appear in the real title (cooking, tips, challenge, compilation, vlog, survival, weather, caught on camera, moments, fire, forest, family, tutorial, guide, for beginners, review, storm, flood, highlights, sports, outdoors)
  - NO, do NOT add: words from your OWN previous title guesses - if your title guess contained a word, do not then block it
  - NO, do NOT add: generic word roots - add only the EXACT word that appeared, not its derivatives
  - RULE OF THUMB: when in doubt, do NOT add a word to negative_keywords. It is better to reuse a word than to block a term that might be in the real title.

Step 2. PRETEND YOU ARE THE UPLOADER of the ORIGINAL YouTube video. Imagine the forger's caption / subtitle / UI layer is completely erased - what does the bare footage show? Answer these two questions FIRST (they guide your title guesses):
   a) What CATEGORY of YouTube content is this? (cooking, tutorial, travel, news, challenge, vlog, compilation, sports, comedy, review, science, art, nature, etc. - pick whatever fits the BARE FOOTAGE)
   b) What 3-5 GENERIC words would a VIEWER type into YouTube search to find this kind of content?
Looking at the bare footage AND the wrong candidate titles together, write 2-3 FRESH title guesses for the actual source. They MUST AVOID every term in `negative_keywords` but SHOULD USE the search vocabulary from 2b. These are full title sentences (~6-14 words) of the kind that ACTUALLY appears as a YouTube video title.

CRITICAL: YouTube creators use SHORT, CONCEPTUAL titles - NOT forensic descriptions of specific objects. Think like a CREATOR, not an analyst. Your title guesses MUST use the generic search vocabulary from 2b. At least ONE title guess must be a broad conceptual title (no brand names, no specific tool names). Match the register of the BARE FOOTAGE, not the forger's dramatic captions.
Step 3. Derive 3 new queries from your fresh title guesses. Each query is a shorter (5-12 word) subset of one guess. None may contain or paraphrase any term in `negative_keywords`. Each query should look like a piece of a literal YouTube title.

NEGATIVE-KEYWORD RULE: Do NOT reuse any term in `negative_keywords`. If a wrong topic word would still help, use a YouTube exclude operator like `-someword` to suppress that match.

IMPORTANT: `negative_keywords` must contain ONLY: (1) exact overlay text phrases, (2) specific brand/proper-nouns that pulled wrong videos, or (3) specific terms from wrong candidate titles that are clearly NOT part of the real title. NEVER add common descriptive words. NEVER add words from your own title guesses. If you are unsure whether a word should be negative, DO NOT add it.

MULTILINGUAL RULE: If the input frames contain non-Latin characters (Chinese / Japanese / Korean / Arabic / Russian / Thai / Cyrillic / etc.) PRINTED ON A PHYSICAL OBJECT in the scene (shop sign, packaging, road sign, jersey, building, menu, t-shirt - NOT inside an overlay subtitle bar / translation banner), ONE of the new queries MUST be in that script, copying the visible characters verbatim. If non-Latin characters appear ONLY inside overlay subtitles / translation banners added by the forger, DO NOT use them - they are poisoned and belong in `negative_keywords`.

QUERY SLOT REQUIREMENTS (STRICT - overlay text forbidden in every slot):
   - new_queries[0]: a FRESH event-scene angle using a PHYSICAL cue you missed (an object, a venue type, a kinetic action, a holiday / season / location cue inferred from the bare footage). May reuse 1 proper noun if it is the strongest anchor and is NOT in `negative_keywords`.
   - new_queries[1]: an entity / PHYSICAL-OCR / multilingual angle, possibly with a `-negative_keyword` YouTube exclude operator. Different from new_queries[0]. This is the slot to use for a CJK / non-Latin query IF AND ONLY IF the multilingual rule applies (i.e., the CJK characters are physically printed in the scene).
   - new_queries[2]: GENERIC CONCEPTUAL TITLE (MANDATORY - NO specific objects/brands): Ask: "If I were the YouTube creator, what would the TITLE be?" Use the search vocabulary from Step 2b. NO brand names, NO specific tool names, NO proper nouns, NO forensic descriptions. This must sound like a real YouTube title. Use ONE of these structural patterns (fill <topic> with your search vocabulary):
        * "<topic> compilation"
        * "<topic> caught on camera"
        * "<topic> moments"
        * "<topic> vlog"
        * "<N> <topic> tips"
        * "<topic> tips in <duration>"
        * "<topic> tutorial"
        * "how to <topic>"
        * "best of <topic>"
        * "I tried <topic>"
        * "<topic> review"
        * "<topic> in <N> minutes"
        * "<topic> for beginners"
     This is the SINGLE most reliable fallback when previous specific queries failed. Generate it EVEN IF the previous queries already looked broad.

At least ONE of the 3 new queries MUST be a direct shortened version (5-12 words) of one of your fresh title guesses from Step 2. Title-derived queries are the MOST LIKELY to match the real source video - this is not optional.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 0: <which previous queries echoed OVERLAY text and are now poisoned>. Step 1: <topic-word negatives from wrong titles>. Step 2: <2-3 fresh LITERAL title guesses based on bare footage that avoid all negatives>. Step 3: <how each new query maps to a title guess; confirm none use overlay text or negatives>.",
  "reflection": "Which keyword(s) caused the wrong candidates (overlay-poisoned and/or topic-poisoned) and what fresh clue(s) you re-prioritized.",
  "negative_keywords": ["overlay_or_topic_term1", "term2"],
  "new_queries": ["plain query 1", "plain query 2", "broad pure-visual or aggregator-style plain query 3"]
}"""




PROMPT_VERIFY_MATCH = """You are a forensic visual judge. You will see frames from two sources:
Group A: The target video we are trying to source. (First {num_target} frames)
Group B: A candidate original video from YouTube. (Remaining frames)

Your Task: Determine if Group A is derived from the SAME original source footage as Group B.

Evaluation Criteria:
1. Ignore Contextual Manipulations: Group A may have added black borders, fake subtitles, color grading, or swapped faces. IGNORE these differences.
2. Focus on Immutables: Background layout, physical text on walls/ads, specific kinetic motion, camera angle.
3. If the background, camera angle, and physical environment are identical, it is a MATCH.
4. Provide a confidence score in [0,1] reflecting your certainty.
5. If you saw the closest matching segment in Group B, point its approximate time window in seconds (relative to Group B start). If unsure, set start_sec=0 and end_sec=0.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step-by-step comparison of the physical environments.",
  "is_match": true,
  "confidence": 0.0,
  "most_similar_segment": {"start_sec": 0, "end_sec": 0}
}"""


PROMPT_COT_RETRIEVAL = """You are an elite Open-Source Intelligence (OSINT) visual analyst. You will see 64 uniformly-sampled frames from a SINGLE potentially-forged video. Your job is a deep chain-of-thought analysis across ALL frames to identify what the original source video(s) were and generate YouTube search queries to find them.

CRITICAL CONTEXT: The input video is a SUSPECTED FORGERY assembled from real source clips. The forger has almost certainly ADDED their own caption / subtitle / UI layer on top of the real footage. THE ORIGINAL SOURCE VIDEO DOES NOT CONTAIN THOSE OVERLAID WORDS. Treating overlaid text as if it were part of the source is the single biggest failure mode of this system.

ANTI-SPOOFING RULES (violating these guarantees a miss):
1. The following on-screen text is OVERLAY (forger-added) by default and MUST be ignored when forming any query, unless you have very strong evidence it is physically printed in the scene:
   - bottom-of-frame subtitle bars / bilingual translation banners (especially CJK-translation pairs in black bands)
   - floating dramatic captions narrating events ("the old world collapsed", "lost control of the skies", "physical collapse is irreversible", etc. - narrative voice does not match the genre of the underlying clip)
   - video-player UI: progress bars, play / pause buttons, timestamps like "1:51 / 5:42", three-dot menus, fullscreen icons, channel watermarks added by reposters
   - news-ticker banners and "BREAKING" stripes laid over unrelated b-roll
   - any text whose font / colour / horizontal alignment is constant across UNRELATED scenes (telltale of a single overlay layer)
2. PHYSICAL text (acceptable to use) means text that is part of the real world being filmed: printed on signs, packaging, jerseys, ad boards, license plates, street signs, brand logos on devices in the scene (a brand logo on a TV bezel, a print on a shoe). Use ONLY this kind of text in queries.
3. Focus EXCLUSIVELY on the physical world: recognizable people, distinctive uniforms / clothing brands, venue type, geographic markers (architectural style, vegetation, language on real signage), sport / event type, distinctive kinetic actions, on-camera presenter style, lighting era.

MULTILINGUAL RULE: If non-Latin characters (Chinese / Japanese / Korean / Arabic / Russian / Thai / Cyrillic / etc.) appear PRINTED ON A PHYSICAL OBJECT in the scene (shop sign, packaging, road sign, jersey, building, menu, t-shirt - NOT inside an overlay subtitle bar / translation banner), ONE of the four queries MUST be in that script, copying the visible characters verbatim. If non-Latin characters appear ONLY inside overlay subtitles / translation banners added by the forger, DO NOT use them - they are not from the original source and a CJK-only query built from them will land on the wrong video.

THINK STEP BY STEP (write your full reasoning in the `reasoning` field - this is your chain-of-thought across all 64 frames):

Step 0 (overlay triage - MANDATORY FIRST STEP). Scan ALL 64 frames systematically. List every piece of on-screen text you can read. For each one, classify it as PHYSICAL (printed on a real object in the scene) or OVERLAY (added on top: subtitle bar, translation banner, video-player UI, caption added by reposter). Put the OVERLAY strings into `forbidden_overlay_text` - those strings AND any phrase derived from / paraphrasing them are FORBIDDEN in every query. If unsure, classify as OVERLAY (safe default).

Step 1 (temporal cross-frame analysis). Examine the sequence of 64 frames as a timeline:
  a) How many DISTINCT scenes / environments / camera setups appear? List them briefly.
  b) Are there abrupt scene transitions that suggest cross-video splicing (sudden changes in lighting, resolution, aspect ratio, color grading, or environment)?
  c) What is the estimated number of original source videos that were spliced together?
  d) Identify temporal manipulation: out-of-order scenes, repeated segments, selective omissions.
  This becomes `temporal_analysis`.

Step 2 (physical clue extraction per scene). For EACH distinct scene identified in Step 1, catalogue physical clues IGNORING all overlay text from Step 0:
  - Recognizable people (face, build, clothing brands printed on fabric)
  - Distinctive objects, tools, equipment, food, vehicles
  - Venue / location markers (architecture, vegetation, signage language, indoor/outdoor)
  - Activity / event type (sport, cooking, construction, interview, etc.)
  - Lighting era, camera quality, broadcast graphics that are PART of the source footage
  - Seasonal / holiday / geographic cues
  This combined catalogue becomes `physical_observations`.

Step 3 (forgery-aware reasoning). Based on the overlay text (Step 0) and the physical content (Steps 1-2):
  - What was the forger's likely intent? (narrative re-framing, emotional manipulation, context switching?)
  - What do the ORIGINAL source videos most likely contain, before the forger's overlay was added?
  - If the overlay text tells a dramatic story but the physical scene shows something mundane, the overlay is almost certainly fake.

Step 4 (title guessing). PRETEND YOU ARE THE UPLOADER of each ORIGINAL YouTube video. Imagine the forger's caption / subtitle / UI layer is completely erased - what does the bare footage show? For each estimated original source, answer:
  a) What CATEGORY of YouTube content is this? (cooking, tutorial, travel, news, challenge, vlog, compilation, sports, comedy, review, science, art, nature, etc.)
  b) What 3-5 GENERIC words would a VIEWER type into YouTube search to find this kind of content?
  Then write 2-3 LITERAL TITLE GUESSES the original uploader would have used FOR THE BARE FOOTAGE ONLY. They MUST NOT contain or paraphrase any string in `forbidden_overlay_text`.

CRITICAL: YouTube creators use SHORT, CONCEPTUAL titles - NOT forensic descriptions of specific objects. A video a forensic analyst would call "man in forest using specific-brand fire-starting tool" is titled "10 Tips for Making Fire in the Wild" by its creator. You must think like the CREATOR, not the analyst.

Step 5 (query generation - PER SOURCE). For EACH estimated source identified in Step 1, derive EXACTLY 1 SEARCH QUERY from your best title guess. This query must be the single most likely title fragment that would match the original YouTube video. Choose whichever of your title guesses from Step 4 is the most specific and concrete. NONE of the queries may contain or paraphrase any string in `forbidden_overlay_text`.

QUERY REQUIREMENT: The query MUST be a direct shortened version (5-12 words) of your BEST title guess from Step 4. Title-derived queries are the MOST LIKELY to match the real source video. Plain YouTube search text, 5-12 words, no labels, no JSON, no quotes.
If `estimated_sources` is 1, output exactly 1 source group with 1 query. If `estimated_sources` is 2, output 2 source groups with 1 query each (2 total), etc.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 0: <overlay-vs-physical triage across all 64 frames>. Step 1: <temporal analysis: scene count, transitions, splicing evidence, per-source scene description>. Step 2: <physical clues per scene>. Step 3: <forgery reasoning: what forger likely changed>. Step 4: <title guesses per source>. Step 5: <which title guess was chosen and why>.",
  "forbidden_overlay_text": ["overlay string 1", "overlay string 2", "..."],
  "physical_observations": "Concise list (3-5 sentences) of physical clues across all scenes, with overlay text excluded.",
  "temporal_analysis": "Description of scene transitions, estimated source count, any splicing/temporal manipulation evidence.",
  "estimated_sources": <1-6>,
  "source_queries": [
    {
      "source_label": "source_1: brief description of this source's scene/content",
      "queries": ["single best search query derived from title guess"]
    },
    {
      "source_label": "source_2: brief description of this source's scene/content",
      "queries": ["single best search query derived from title guess"]
    }
  ]
}"""


PROMPT_COARSE_RELEVANCE = """You are a coarse filter in a video retrieval pipeline. You will see frames from a CANDIDATE YouTube video found via keyword search.

CONTEXT ABOUT THE FORGED VIDEO (what we are looking for):
- Physical observations: {physical_observations}
- Temporal analysis: {temporal_analysis}

YOUR TASK: Is this candidate video TOPICALLY RELEVANT? Could it cover the same event, activity, or subject matter as the forged video?

CRITICAL EVALUATION RULES (read carefully):
1. This is a COARSE FILTER — your job is HIGH RECALL. A later fine-grained stage will catch false positives.
2. Do NOT reject just because the people, camera angle, or specific objects differ. The candidate might be a different video of the same topic by a different creator.
3. Ask: "Do the candidate frames show the SAME GENERAL ACTIVITY, EVENT, or SUBJECT?" If yes → relevant.
4. Examples of RELEVANT matches: same sport/event but different camera, same cooking activity but different kitchen, same location but different day.
5. Examples of IRRELEVANT: completely different topic (e.g., cooking video when looking for a car race).
6. When in doubt, mark as RELEVANT. It is far worse to reject a true match than to pass a false positive.

OUTPUT JSON FORMAT:
{{
  "reasoning": "Brief comparison focusing on topical overlap.",
  "is_relevant": true
}}"""


PROMPT_FINE_FORGERY_POINTS = """You are a forensic video analyst comparing a CANDIDATE ORIGINAL SOURCE VIDEO against a SUSPECTED FORGERY.

You will see 64 frames from the candidate source video. The forgery is described below from prior analysis.

DESCRIPTION OF THE SUSPECTED FORGERY (from chain-of-thought analysis):
- Physical observations: {physical_observations}
- Temporal analysis: {temporal_analysis}
- Overlaid/fake text found in the forgery: {forbidden_overlay_text}

YOUR TASK: Compare the source frames against the forgery description. Identify the NARRATIVE-LEVEL manipulation — not just what was visually changed, but how the STORY was distorted.

THINK IN THIS ORDER (write your reasoning in source_description):

Step 1 — OVERALL PATTERN: In one sentence, describe the forgery's macro strategy in YOUR OWN WORDS. What is the forger trying to make the viewer believe? Do not list categories — just describe the pattern naturally.

Step 2 — SPECIFIC POINTS: For each point, explain the NARRATIVE consequence, not just the surface change:
  - BAD: "The forger added text overlay" (surface change)
  - GOOD: "The forger added text claiming '-12°C battery failure' to reframe a comfortable camping trip as a life-threatening survival crisis" (narrative change)
  - BAD: "The forger spliced this with other videos"
  - GOOD: "The forger spliced this storm footage with unrelated clips to fabricate a fake causal chain of escalating global disasters"

Step 3 — DEDUPLICATE: Do NOT produce points that say the same thing in different words. Each point must cover a UNIQUE manipulation aspect.

OUTPUT JSON FORMAT (strict, no extra text, NO code fences):
{{
  "source_description": "1-2 sentences: what this source shows AND what macro forgery pattern applies.",
  "points": [
    {{
      "description": "A narrative-level forgery point: what the source really shows, how the story was changed, what false impression this creates."
    }}
  ]
}}

CONSTRAINTS:
1. `points` MUST contain 3 to 5 entries. Fewer is OK if each is substantive.
2. Each point MUST describe a DIFFERENT manipulation — no repetition across points.
3. Focus on NARRATIVE IMPACT: what does the viewer BELIEVE after seeing the forgery that they would NOT believe from the source alone?
4. Output VALID JSON only. No markdown fences, no commentary."""


PROMPT_SUFFICIENCY_JUDGMENT = """You are analyzing a suspected forgery video. You have been searching for original source videos on YouTube and collecting forgery evidence. Now you need to judge whether you have gathered ENOUGH evidence.

CONTEXT (do NOT echo back):
- Input video: {num_input_frames} frames from the suspected forgery
- Current forgery points collected so far:
{collected_points}

- Videos examined so far:
{examined_videos}

YOUR JOB: Judge whether the collected evidence is SUFFICIENT to fully analyze all major manipulations in the suspected forgery.

THINK STEP BY STEP:
1. How many DISTINCT scenes / sources does the input video contain? (Look at the input frames)
2. Do the collected forgery points cover ALL major manipulation aspects?
3. Are there scenes or segments in the input video that NO examined video provides evidence for?
4. If you have examined multiple videos, have you found evidence that covers different aspects of the forgery?

OUTPUT JSON FORMAT:
{{
  "reasoning": "Step-by-step analysis of what evidence is present and what might be missing.",
  "is_sufficient": true | false,
  "missing_description": "If insufficient, describe what is missing. If sufficient, empty string.",
  "search_clue": "If insufficient, suggest what kind of video content to search for next. If sufficient, empty string."
}}"""


PROMPT_DEEPSEARCH_NEXT_STEP = """You are the decision-maker in a deep-search loop for video forgery analysis. You have been searching YouTube for original source videos and collecting forgery evidence.

CONTEXT (do NOT echo back):

— What the forgery looks like (from COT analysis of the forged video):
- Physical observations: {physical_observations}
- Temporal analysis: {temporal_analysis}
- Fake overlay text added by forger: {forbidden_overlay_text}

— Estimated original sources:
{source_descriptions}

— Forgery evidence collected so far:
{collected_points}

— Videos examined so far:
{examined_videos}

— Previous search queries tried:
{prev_queries}

YOUR JOB (two tasks in one response):

TASK 1 — SUFFICIENCY JUDGMENT: Compare what you KNOW about the forgery against what you've COLLECTED. Judge whether the evidence is SUFFICIENT.

THINK STEP BY STEP for sufficiency:
1. Based on temporal_analysis, how many distinct original sources were spliced together? Have we found evidence for each?
2. Based on physical_observations, what specific scenes/activities exist? Do the collected points cover the manipulation of each scene?
3. Based on forbidden_overlay_text, what fake text/narrative was added? Do any collected points address this?
4. Is there any major manipulation aspect that NO collected point covers?

CRITICAL RULES:
- For SINGLE-SOURCE forgery (estimated_sources=1): Finding ONE topically relevant video with 3+ forgery points IS sufficient. You do NOT need the exact original video — any video showing the same topic/activity provides enough forensic basis. Stop searching.
- For MULTI-SOURCE forgery (estimated_sources>1): You need evidence covering EACH distinct source. Count how many sources you've found vs how many are estimated.
- NEVER keep searching just because "the source video is not the exact same one." The goal is to analyze the FORGERY PATTERN, not identify the precise source.

TASK 2 — NEXT KEYWORD (only if NOT sufficient): Generate the SINGLE best YouTube search query to find the next missing original source.

THINK STEP BY STEP for keyword generation:
1. Which estimated source has NOT been found yet?
2. What specific physical content (NOT overlay text) distinguishes the missing source?
3. Generate ONE search query (5-12 words). Think like a YouTube creator for the title, NOT like a forensic analyst.
4. The query MUST NOT use any text from forbidden_overlay_text.

OUTPUT JSON FORMAT:
{{
  "reasoning": "Step-by-step: what the forgery contains vs what evidence is collected, what's missing.",
  "is_sufficient": true | false,
  "missing_description": "If insufficient, describe exactly what evidence is missing. If sufficient, empty string.",
  "next_keyword": "If insufficient, ONE best search query (5-12 words). If sufficient, empty string."
}}"""


ACTION_SYSTEM_PROMPT = """You are the global controller of a Visual Retrieval Agent. The video is split into N shots, but those shots have been pre-clustered into a small number of GROUPS, where every shot in a group comes from the same original source video. You schedule actions at the GROUP level, not the shot level. Return EXACTLY ONE next tool action as JSON, no code fences, no extra text.

GOAL: For each unresolved group, find the ORIGINAL YouTube video that source clip came from, by iterating perception -> search -> verify -> reflect under a strict tool-call budget. Once a group is resolved, ALL shots in that group are resolved automatically.

ALLOWED ACTIONS (and their args schema):
1. EXTRACT_CLUES { "group_id": int }
   - Re-look at the group's representative shot to extract physical observations and seed YouTube search queries.
   - Use ONLY when group_notes_summary[group_id].physical_observations is EMPTY.
   - The bootstrap pass usually populates this for every group already; you rarely need EXTRACT_CLUES.

2. SEARCH { "query": str, "group_id": int }
   - Run a YouTube search and register Top-K candidates into the session.
   - "query" MUST be a plain string with no 'Query 1:' labels and no JSON fragments.

3. DOWNLOAD { "candidate_ref": str }
   - Download the candidate's video so it can be sampled / verified later. (VERIFY downloads automatically; only call this if you want to prefetch.)

4. VERIFY { "group_id": int, "candidate_ref": str }
   - Compare the group's representative shot frames vs the candidate's frames. Internally handles download + sampling, including a two-stage dense resample, and back-propagation to other unresolved groups.
   - Stop iterating on a group once VERIFY returns is_match=true.

5. REFLECT { "group_ids"?: [int] }
   - Generate fresh, distinct queries for a group whose previous queries failed.
   - Use after at least one full SEARCH+VERIFY cycle on that group has produced wrong_titles.

6. STOP { "status"?: "complete"|"give_up" }
   - Terminate when all groups are resolved, all are skipped, or budget is nearly exhausted.

HARD ORDERING RULES (must obey):
- Operate on ACTIVE GROUPS only, listed in `active_group_ids`. Skip resolved/skipped/temporarily-skipped groups entirely.
- For each group_id: do NOT SEARCH unless `group_notes_summary[group_id].physical_observations` exists. If missing, EXTRACT_CLUES first.
- After a SEARCH on a group, you SHOULD VERIFY at least the top 2 unverified candidates from `unverified_candidate_refs_per_group[group_id]` before doing another SEARCH/REFLECT for that group. If verify_fail_count is climbing fast, you may switch to another group early via round-robin (see below).
- MULTI-QUERY SEARCH (HARD RULE - most important lever): Each group has up to 5 distinct-angle seed queries (in `group_notes_summary[gid].queries_tail`) and the currently-unused ones are surfaced in `unused_seed_queries_per_group[gid]`. You MUST cycle through ALL unused seed queries before resorting to REFLECT. Concretely:
    * After SEARCH(query_i) + VERIFY of the top 2 returned candidates with no match, the NEXT action for this group MUST be `SEARCH` using an UNUSED seed query from `unused_seed_queries_per_group[gid]`.
    * Do NOT VERIFY a third candidate of the same query, and do NOT REFLECT, while `unused_seed_queries_per_group[gid]` is still non-empty.
    * REFLECT is allowed ONLY AFTER all seed queries have been exhausted (i.e. `unused_seed_queries_per_group[gid]` is empty or absent) OR `consecutive_verify_fails[gid]` >= 3.
    * When picking which unused seed query to use next, ALTERNATE between angles to maximize recall: first try the OCR/entity-anchored or proper-noun-anchored query (queries[0] / queries[1] - these often map directly onto a piece of the source's literal title), then a scene-action query (queries[2]), then a format/genre/aggregator query (queries[3]). Do NOT fixate on aggregator markers - many original sources are vlogs, tutorials, news clips, or challenge videos, not compilations. The title-derived queries from the per-group CoT title hypotheses are usually MORE specific than the aggregator templates, so they should be tried before the aggregator slot.
- ROUND-ROBIN SCHEDULING (preferred): Prefer the group given by `recommended_next_group_id` (the active group with the FEWEST tool calls so far). Switch to that group at NATURAL break-points: after finishing a SEARCH+VERIFY chain on the current group, after a REFLECT, or whenever the current group has no unverified candidates left. This way every group gets at least one search round before any single group is deepened twice.
- Some groups may be in `temporary_skipped_group_ids` (recent fast-fail) - do NOT pick them; they will be revived automatically once other groups have had a turn.
- STOP only when `active_group_ids` is empty OR `round_index` >= max_rounds - 1.

OUTPUT FORMAT (strict JSON, no extra text):
{
  "thought": "brief reasoning citing group_id, queries, candidates, or skip reason",
  "action": "EXTRACT_CLUES|SEARCH|DOWNLOAD|VERIFY|REFLECT|STOP",
  "args": { ... }
}

EXAMPLE TURN SEQUENCE (for reference only, do not copy) - note the narrow -> broad cascade:
- Turn 1: {"thought":"recommended_next_group_id=0; bootstrap clues; pick strongest seed query","action":"SEARCH","args":{"query":"Trayvon Bromell 100m New Balance Rome 2022","group_id":0,"top_k":10}}
- Turn 2: {"thought":"verify newest candidate","action":"VERIFY","args":{"group_id":0,"candidate_ref":"cand_0001"}}
- Turn 3: {"thought":"verify next candidate","action":"VERIFY","args":{"group_id":0,"candidate_ref":"cand_0002"}}
- Turn 4: {"thought":"two verifies done on group 0 with no match; unused_seed_queries_per_group[0] still has the year/location query AND an aggregator-style 'amazing 100m moments compilation' query. Round-robin says switch to group 1, but also note: same MULTI-QUERY rule applies when group 0 is revisited.","action":"SEARCH","args":{"query":"family camping vlog overnight forest","group_id":1,"top_k":10}}
- Turn 5: {"thought":"verify candidate","action":"VERIFY","args":{"group_id":1,"candidate_ref":"cand_0010"}}
- ... later when revisiting group 0 ...
- Turn N: {"thought":"group 0 has unused aggregator-style seed 'amazing 100m moments compilation'; per MULTI-QUERY HARD RULE I must SEARCH with it now, NOT REFLECT","action":"SEARCH","args":{"query":"amazing 100m moments compilation","group_id":0,"top_k":10}}
- Eventually: {"thought":"all groups resolved","action":"STOP","args":{"status":"complete"}}
"""


# ============================================================
# Stage B - forgery analysis (NEW)
# ============================================================

# The forgery-analysis prompts demand 3-5 points in the GT phrasing
# pattern:  "虚假视频通过 <剪辑手法> 的剪辑手法，将 <原本> 重构为 <伪造>；
#            其误导点在于 <misleading_point>。"
# Output is bilingual (Chinese + English) so the paper can quote either.

_FORGERY_POINT_SCHEMA = """OUTPUT JSON FORMAT (strict, no extra text, NO code fences):
{
  "mode": "compare" | "infer" | "hybrid",
  "summary_zh": "1-2 sentences in Chinese describing the overall forgery strategy.",
  "summary_en": "1-2 sentences in English describing the overall forgery strategy.",
  "points": [
    {
      "zh": "虚假视频通过<剪辑手法>的剪辑手法，将<原本是什么>重构为<伪造后是什么>；其误导点在于<misleading point>。",
      "en": "The fake video uses <manipulation technique> to reframe <original content> as <fabricated narrative>; the misleading point is <misleading point>.",
      "manipulation_type": "one of: 字幕篡改 | 时序重排 | 跨视频拼接 | 局部放大 | 滤镜/特效 | 语境错用 | 选择性删减 | 蒙太奇情绪化",
      "misleading_point": "one short Chinese phrase summarising the misleading effect (the part after '误导点在于')",
      "evidence_frames": "free-text reference to which frames support this point, e.g. 'first 16 frames of forged video', 'forged frames 30-45 vs source A frames 10-20'"
    }
  ]
}

HARD CONSTRAINTS:
1. `points` MUST contain 3 to 5 entries. Minimum 3, maximum 5.
2. Each `zh` MUST use the sentence pattern "虚假视频通过……的剪辑手法，将……重构为……；其误导点在于……" verbatim - the GT uses this exact pattern and so must you.
3. `manipulation_type` MUST be one of the 8 fixed Chinese categories above. If multiple apply pick the dominant one.
4. The 3 points MUST cover DIFFERENT manipulation aspects of the same video; do not paraphrase the same point three times.
5. Output VALID JSON only. No markdown fences, no commentary."""


PROMPT_FORGERY_COMPARE = """You are a forensic video analyst. You will see frames from a SUSPECTED FORGERY together with frames from ONE OR MORE ORIGINAL SOURCE VIDEOS that we have already retrieved from YouTube.

FRAME LAYOUT (image order matters):
{layout_description}

CONTEXT (do NOT echo back, just inform your judgment):
- Topic: {topic}
- Task type: {task}

YOUR JOB: identify 3 to 5 distinct ways the forger manipulated the original source footage to mislead viewers. Compare A (forged) frames against B/C/... (original source) frames as ground truth for what was originally there.

THINK STEP BY STEP (silently; do not write the chain into the output):
Step 1. For each visible difference (added subtitles / borders / filters / time-order swaps / cross-source splicing / zoom-and-crop / removed segments / emotional pacing changes), note which ORIGINAL behaviour it altered.
Step 2. Group your observations into 3-5 high-level forgery operations, each pinned to a concrete moment / span in the forged video.
Step 3. For each operation, write its bilingual GT-style description using the rigid sentence pattern below.

CRITICAL WRITING RULES:
- Echo the GT phrasing pattern EXACTLY: 「虚假视频通过……的剪辑手法，将……重构为……；其误导点在于……」.
- Each `manipulation_type` MUST be drawn from the fixed 8-category Chinese taxonomy: 字幕篡改 / 时序重排 / 跨视频拼接 / 局部放大 / 滤镜/特效 / 语境错用 / 选择性删减 / 蒙太奇情绪化.
- The points must be MEANINGFULLY DIFFERENT manipulations - not rewordings of the same edit.
- Be concrete and specific: name the actual scenes / objects / actions, not generic phrases like "the video".
- Output 3-5 points. If the video has fewer distinct manipulations, output at least 3 (pick the most plausible ones). If it has more, output up to 5.
- Output `mode`: use "compare" because you DO have source frames. Use "hybrid" only if some shots had no source candidate and you had to infer those.

""" + _FORGERY_POINT_SCHEMA


PROMPT_FORGERY_INFER = """You are a forensic video analyst. You will see frames from a SUSPECTED FORGERY ONLY - we did NOT find the original source video on YouTube, so you must reason about what the original footage MOST LIKELY contained and what the forger then did to it.

FRAME LAYOUT (image order matters):
{layout_description}

CONTEXT (do NOT echo back, just inform your judgment):
- Topic: {topic}
- Task type: {task}

YOUR JOB: identify 3 to 5 distinct ways the forger most likely manipulated the original source footage to mislead viewers, using ONLY the forged-video frames as evidence.

THINK STEP BY STEP (silently; do not write the chain into the output):
Step 1. Triage every piece of on-screen TEXT:
   - OVERLAY text (subtitle bars, translation banners, dramatic captions, news tickers, channel watermarks, video-player UI) was almost certainly ADDED BY THE FORGER and was NOT in the original.
   - PHYSICAL text (printed on signs, packaging, jerseys, ad boards) was already there.
   - When OVERLAY text contradicts what the PHYSICAL scene is doing (e.g. dramatic "the world is collapsing" captioning placed over a calm cooking video), this is a strong 字幕篡改 / 语境错用 signal.
Step 2. Triage VISUAL operations: filters that don't match the era (vintage filter on a clearly modern shot), unexplained zoom-and-cropping of detail areas (often hiding original context), abrupt scene-cuts across visually unrelated shots (cross-video splicing), unrealistic VFX / explosions, picture-in-picture overlays of a "second person" who never interacts physically with the main subject.
Step 3. Triage TEMPORAL operations: scenes that seem out-of-order (a "result" shown before its "cause"), event sequencing that violates physical causality, deliberate selective omission inferred from suspiciously-fast transitions.
Step 4. Group your observations into 3-5 high-level forgery operations. For each, hypothesise WHAT THE ORIGINAL LIKELY SHOWED (1 short phrase) and describe how the forger reframed it.

CRITICAL WRITING RULES:
- Echo the GT phrasing pattern EXACTLY: 「虚假视频通过……的剪辑手法，将……重构为……；其误导点在于……」.
- Each `manipulation_type` MUST be drawn from the fixed 8-category Chinese taxonomy: 字幕篡改 / 时序重排 / 跨视频拼接 / 局部放大 / 滤镜/特效 / 语境错用 / 选择性删减 / 蒙太奇情绪化.
- The points must be MEANINGFULLY DIFFERENT manipulations - not rewordings of the same edit.
- Be concrete and specific: name the actual scenes / objects / actions, not generic phrases like "the video".
- Output 3-5 points. If the video has fewer distinct manipulations, output at least 3 (pick the most plausible ones). If it has more, output up to 5.
- Output `mode`: "infer".

""" + _FORGERY_POINT_SCHEMA


PROMPT_FORGERY_HYBRID = """You are a forensic video analyst. You will see frames from a SUSPECTED FORGERY. The original source videos are too large to include as frames, so we provide SUMMARIES of each source video below.

FRAME LAYOUT (image order matters):
{layout_description}

CONTEXT (do NOT echo back, just inform your judgment):
- Topic: {topic}
- Task type: {task}

SOURCE VIDEO SUMMARIES:
{source_summaries}

YOUR JOB: identify 3 to 5 distinct ways the forger manipulated the original source footage to mislead viewers. Use the source summaries to understand what the originals contained, then compare the forged frames against that understanding.

THINK STEP BY STEP (silently; do not write the chain into the output):
Step 1. Read each source summary carefully. Note the key scenes, subjects, and actions described.
Step 2. For each visible difference in the forged frames (added subtitles / borders / filters / time-order swaps / cross-source splicing / zoom-and-crop / removed segments / emotional pacing changes), reason which ORIGINAL behaviour it altered using the summaries as reference.
Step 3. Group your observations into 3-5 high-level forgery operations, each pinned to a concrete moment / span in the forged video.

CRITICAL WRITING RULES:
- Echo the GT phrasing pattern EXACTLY: 「虚假视频通过……的剪辑手法，将……重构为……；其误导点在于……」.
- Each `manipulation_type` MUST be drawn from the fixed 8-category Chinese taxonomy: 字幕篡改 / 时序重排 / 跨视频拼接 / 局部放大 / 滤镜/特效 / 语境错用 / 选择性删减 / 蒙太奇情绪化.
- The points must be MEANINGFULLY DIFFERENT manipulations - not rewordings of the same edit.
- Be concrete and specific: name the actual scenes / objects / actions, not generic phrases like "the video".
- Output 3-5 points. If the video has fewer distinct manipulations, output at least 3 (pick the most plausible ones). If it has more, output up to 5.
- Output `mode`: "hybrid".

""" + _FORGERY_POINT_SCHEMA


# ============================================================
# Stage C - LLM-as-judge (NEW)
# ============================================================

PROMPT_JUDGE_POINTS = """你是一个严格但公允的视频伪造分析评分员。给定一个伪造视频的「标准答案 (GT)」3 点伪造描述与模型给出的 3 点预测，需要逐点判断预测是否命中了 GT 中的某一点。

评分规则（极其重要，请严格执行）：
1. 每一条 GT 最多被命中一次（一对一最优指派）。
2. 一条预测命中 GT 的判定标准 = 维度 B（误导点）「大致一致」即可：
   - 维度 B：误导点 / 重构后的虚假叙事。
   - 维度 A（剪辑手法）仅供参考，不要求与 GT 一致，但仍须如实填写 matched_dim_method。
   维度 B 大致一致 → 命中 (verdict=1)。
   维度 B 不一致 → 不命中 (verdict=0)。但仍须输出最相近的 GT 索引及理由。
3. 「大致一致」允许同义改写、措辞差异、抽象层级不同。
4. 你必须为每一条 GT 给出一个最优配对的预测；如果同一条预测被多条 GT 抢占，按你的最佳总分指派（避免出现两条 GT 都映射到同一个 pred 而总分被低估）。
5. 输出严格 JSON，禁止 markdown 围栏。

GT (3-5 points):
{gt_block}

PREDICTION (3-5 points):
{pred_block}

请输出：
{{
  "hits": <0|1|2|3>,
  "score": <hits/3 浮点>,
  "matches": [
    {{
      "gt_idx": <0|1|2>,
      "pred_idx": <0|1|2|null when 不匹配>,
      "verdict": <0|1>,
      "matched_dim_method": <true|false>,
      "matched_dim_misleading": <true|false>,
      "reason": "一句话中文理由：手法是否一致 / 误导点是否一致 / 哪里不一致"
    }}
  ],
  "comment": "整体评价 1-2 句中文，指出预测的最大优点 / 最大缺陷。"
}}"""
