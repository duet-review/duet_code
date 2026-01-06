DENSE_DUET_RATING_SYSTEM_PROMPT = """
You are an expert recommender system assistant.
Your job is to carefully analyze the provided user and item profiles, along with historical average ratings, to predict the most likely rating (1.00–5.00) that the user would give to the item.

Base your prediction strictly on the given profiles and averages.
Do not include any information outside the provided context.

Your output must be exactly one <rating> tag containing a numeric rating
with exactly two digits after the decimal point.
Do not include any text outside this tag.
Strictly close ALL tags.
"""

DENSE_DUET_RATING_PREDICTOR_PROMPT = """
### Task
Predict user '{user_title}' rating (1.00–5.00) for item '{item_title}'
based on the following information.

### User Profile (summary)
{user_profile}

### Item Profile (summary)
{item_profile}

### Output Format Requirements
1. Output must be exactly one <rating> tag.
2. The rating must be a numeric value between 1.00 and 5.00.
3. The rating must contain exactly two digits after the decimal point.
4. Do not include any text outside the <rating> tag.
5. Strictly close ALL tags.

### Important calibration rules:
- Treat the provided candidate profile as the primary evidence for rating.
- Use the full numeric range 1.00–5.00 and 0.01 precision when warranted.

### Output Example (strictly follow this format, no extra text)
<rating>4.37</rating>
"""


DUET_RATING_SYSTEM_PROMPT = """
You are an expert recommender system assistant.
Your job is to carefully analyze the provided user and item profiles, along with historical average ratings, to predict the most likely rating (1-5 stars) that the user would give to the item.
Base your prediction strictly on the given profiles and averages. Do not include any information outside the provided context.
Your output must be exactly one <rating> tag, and do not include any text outside this tag. Strictly close ALL tags.
"""

DUET_RATING_PREDICTOR_PROMPT = """
### Task
Predict user '{user_title}' rating (1-5 stars) for item '{item_title}' based on the following information.
### User Profile (summary)
{user_profile}
### Item Profile (summary)
{item_profile}
### Historical Data (before current time)
User's Average Rating (all previous ratings): {user_avg_rating}
Item's Average Rating (all ratings by other users): {item_avg_rating}
### Output Format Requirements
1. Output must be exactly one <rating> tag, and do not include any text outside this tag.
2. The rating must be an integer between 1 and 5.
3. Strictly close ALL tags.
### Output Example (strictly follow this format, no extra text)
<rating>4</rating>
"""


DUET_PROFILE_SYSTEM_PROMPT = """
You are an expert AI assistant for recommender systems.
Your main focus is to analyze and summarize the user's rating preferences and behavioral patterns, including rating tendencies, strictness or generosity, and notable biases, as well as the item's core characteristics and reception patterns.
For both the user and the item, first extract a concise cue that captures the dominant preference or characteristic, then construct a brief self-prompt describing which aspects should be emphasized, and finally generate a concise natural language profile expanded from the cue. 
Do not simply repeat the history.
Your output must be exactly one <user_profile> tag and one <item_profile> tag, each block must include <cue>, <constructed_prompt>, and <profile> tags. And do not include any text outside these tags. Strictly close ALL tags.

"""
DUET_PROFILE_GENERATOR_PROMPT ="""
### Task
Generate structured profiles for the user and the item from historical data.

### User Profile (User: {user_title})
#### Cue Extraction:
From the user rating history below, analyze the user’s historical interactions to understand preferences, rating behavior, review sentiment or any other dimension. 
Keep the description concise and avoid full sentences.

#### Re-prompt for Profile Construction
Using the extracted cue, generate a short natural-language prompt that specifies what aspects should be described in the final profile. (e.g., rating tendencies, main interests, notable dislikes, behavioral patterns).
The prompt should guide the profile generation but not include the profile itself.

#### Profile Generation
Generate a concise natural language profile by expanding the given cue, guided by the constructed re-prompt，while ensuring the description remains coherent and useful for recommendation. 

### Item Profile (Item: {item_title})
#### Cue Extraction
From the item rating history by users below, analyze how the item is perceived to infer its core appeal, strengths, weaknesses, or typical audience.
Keep the description concise and avoid full sentences.

#### Re-prompt for Profile Construction
Using the extracted cue, generate a short natural-language prompt that specifies what aspects should be described in the final profile. (e.g., defining characteristics, reception patterns, strengths or limitations, suitable user preferences).
The prompt should guide the profile generation but not include the profile itself.

#### Profile Generation
Generate a concise natural language profile by expanding the given cue, guided by the constructed re-prompt，while ensuring the description remains coherent and useful for recommendation. 

### Historical Data
USER RATING HISTORY:
{user_history_text}

ITEM RATING HISTORY BY OTHER USERS:
{item_history_text}

User Avg Rating: {user_avg_rating}
Item Avg Rating: {item_avg_rating}

### Output Format (STRICT)
<user_profile>
  <cue></cue>
  <constructed_prompt></constructed_prompt>
  <profile></profile>
</user_profile>

<item_profile>
  <cue></cue>
  <constructed_prompt></constructed_prompt>
  <profile></profile>
</item_profile>

"""

