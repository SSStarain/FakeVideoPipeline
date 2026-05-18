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


PROMPT_COT_RETRIEVAL_V3 = """You are a video forensic analyst. You will see 64 frames uniformly sampled from a video that may contain misinformation.

Your task is to deeply understand the video content, identify key entities and potential logical anomalies, and generate a precise initial search query to help find the original video that can verify or reconstruct the truth in a later retrieval stage.

Think step by step (write your full reasoning in the reasoning field):

== Step 1: Content Understanding ==

Examine the 64 frames as a timeline:

a) What different scenes/environments appear in the video? Briefly describe each scene.

b) Identify the key entities and information in the video:
   - People: Who appears? Can you identify specific individuals, such as celebrities, athletes, or political figures? What are their appearance, clothing, and surroundings?
   - Locations: Where is it? Are there identifiable landmarks, signs, logos, or architectural styles?
   - Events: What is happening? A sports match, variety show, news event, everyday scene?
   - Time: Can any time information be inferred? For example, dates in broadcast graphics, seasonal clues, or the era suggested by equipment.
   - On-screen text: What does all readable text in the frame say, including captions, news headlines, program logos, scoreboards, signs, etc.? What facts do these texts claim or imply?

c) Are there discontinuities between scenes, such as sudden changes in lighting, resolution, or environment, suggesting the footage may come from multiple sources?

== Step 2: Logical Reasoning ==

Combine visual information, text information, and common sense to look for potential logical contradictions or anomalies:

- Are the facts claimed or implied in the video, through visuals, text, or event sequence, consistent with common sense or known facts?
- Are there unreasonable links between people, locations, or times across different scenes?
- Are there events that could not have happened at the same time but appear to be edited together?
- Is there any contradiction between the visual content and the on-screen text?
- If no obvious contradiction is found, explain the key fact that is most worth verifying.

== Step 3: Search Strategy ==

Based on Step 1 and Step 2, determine your first search target:

- What are you going to search for? For example, a person’s real experience, an original report of an event, or the original source of a piece of footage.
- Why choose this as the first search target? For example, it contains the most information, is easiest to verify, or is most likely to match the original video.
- Generate a YouTube search query of 5–12 words.

Query strategy:
- Prioritize specific entities you have identified: people’s names, program names, event names, location names, and dates.
- If you can identify a specific person + event, combine them directly, such as "Messi goal France 2022 World Cup final".
- If you identify a program or competition, use the program name + key person or key plot point, such as "America's Got Talent Jessica Sanchez audition".
- If you cannot identify specific entities, use the content category + the most distinctive visual feature, such as "Bangkok street food night market".
- Key information from on-screen text, such as program names, people’s names, or event names, can be included as part of the search terms.

Output in JSON format:
{
  "reasoning": "Step 1: <content understanding, including scenes, entities, and text information>. Step 2: <logical reasoning, contradictions found or facts that need verification>. Step 3: <search strategy selection and query derivation>.",
  "entities": {
    "people": ["list of identified people"],
    "locations": ["list of identified locations"],
    "events": ["identified events/programs/competitions"],
    "text_claims": ["key facts claimed by on-screen text"]
  },
  "logical_analysis": "Logical analysis of the video content, including any contradictions found or key facts that need verification. If no obvious contradiction is found, explain what is most worth verifying.",
  "physical_observations": "Summarize the most searchable visual clues in 3–5 sentences.",
  "search_intent": "The purpose of this search, such as verifying a person’s identity, finding the original report of an event, or locating the source of the footage.",
  "initial_query": "A search query of 5–12 words"
}"""


PROMPT_COARSE_RELEVANCE = """You are a coarse filter in a video retrieval pipeline.

You will see TWO frame groups:
- Group A: frames from the SUSPECTED FORGERY video.
- Group B: frames from a CANDIDATE YouTube video.

CONTEXT FROM PRIOR COT ANALYSIS:
- Physical observations: {physical_observations}
- Temporal analysis: {temporal_analysis}

YOUR TASK: judge whether Group B is likely from the SAME EVENT / SAME REAL-WORLD SCENE FAMILY as Group A (or a close same-topic source worth keeping).

CRITICAL EVALUATION RULES:
1. This is a COARSE FILTER with HIGH RECALL. Prefer false positives over false negatives.
2. Do NOT rely on pixel-level similarity.
3. Ignore differences caused by compression, bitrate, resolution, color grading, subtitles, black bars, crop, mirror, small timing offset, and camera/viewpoint variation.
4. Focus on semantic anchors: event type, scene layout, recurring objects, role relations, action pattern, and temporal storyline.
5. If Group B could plausibly provide source-side evidence for Group A's narrative, mark relevant.
6. Only mark irrelevant when topics/scenes are clearly different.

OUTPUT JSON FORMAT:
{{
  "reasoning": "Brief semantic comparison between Group A and Group B.",
  "is_relevant": true
}}"""


PROMPT_FINE_FORGERY_POINTS = """You are a forensic video analyst comparing:
- Group A: frames from a SUSPECTED FORGERY video
- Group B: frames from a CANDIDATE source video

PRIOR CONTEXT:
- Physical observations: {physical_observations}
- Temporal analysis: {temporal_analysis}
- Overlaid/fake text found in the forgery: {forbidden_overlay_text}

YOUR TASK: Do a cross-group semantic comparison and extract NARRATIVE-LEVEL forgery points.
This is NOT pixel matching. You must tolerate quality/angle/edit differences and focus on story distortion.

THINK IN THIS ORDER (write reasoning in source_description):
1. OVERALL ALIGNMENT: summarize what real event/activity Group B likely shows, and how it aligns with Group A's underlying footage.
2. MANIPULATION IMPACT: explain how Group A reframes, exaggerates, fabricates context, or changes causality versus Group B.
3. DEDUPLICATE: each point must cover a unique manipulation aspect.

OUTPUT JSON FORMAT (strict, no extra text, NO code fences):
{{
  "source_description": "1-2 sentences: semantic relation between Group A and Group B plus macro forgery pattern.",
  "points": [
    {{
      "description": "Narrative-level forgery point: what Group B indicates, how Group A changed the story, and the false impression created."
    }}
  ]
}}

CONSTRAINTS:
1. `points` should contain 3 to 5 entries when evidence supports it.
2. No duplicate points with paraphrased wording.
3. Ignore pure quality/viewpoint differences unless they support a narrative manipulation claim.
4. Output VALID JSON only. No markdown fences, no commentary."""


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


# ============================================================
# Stage B - forgery analysis (NEW)
# ============================================================

# The forgery-analysis prompts demand 3-5 points in the GT phrasing
# pattern:  "虚假视频通过 <剪辑手法> 的剪辑手法，将 <原本> 重构为 <伪造>；
#            其误导点在于 <misleading_point>。"
# Output is bilingual (Chinese + English) so the paper can quote either.

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
